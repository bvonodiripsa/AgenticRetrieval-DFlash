"""Prompt templates for the Food Knowledge Graph pipeline.

OFFLINE prompts  – used during KG construction (multi-round extraction).
ONLINE  prompts  – used at query time (single LLM call with graph context).
"""

# =============================================================================
# OFFLINE: Initial triple extraction from a food product document
# =============================================================================

INITIAL_EXTRACTION_PROMPT = """You are a knowledge graph builder for a food product catalog.
The source data is a structured JSON document from a product database.

Extract subject-predicate-object triples from ALL fields in this JSON document.

FIELD SEMANTICS:
- "product_id": unique product identifier
- "product_title" / "product_title_translated": the product name (use as primary entity)
- "brand": manufacturer or brand name
- "ingredients" / "ingredients_translated": full ingredient list (parse individual ingredients)
- "allergens" / "allergens_translated": allergen warnings
- "claims" / "claims_translated": dietary/marketing claims (e.g. sugar-free, organic, gluten-free)
- "pack_size" / "pack_size_translated": package size/weight
- "country_code": country of origin/sale

RULES:
- Use the "product_title_translated" (or "product_title" if no translation) as the primary entity name
- Include the product_id in the subject: format as "product_title (product_id: XXXXX)"
- Extract EACH ingredient as a separate triple: (product, has_ingredient, ingredient_name)
- Extract allergens: (product, contains_allergen, allergen) and (product, may_contain, allergen)
- Extract dietary claims: (product, has_claim, claim) — e.g. "sugar free", "gluten free", "organic"
- Infer dietary properties: if sugar-free → (product, free_from, sugar); if vegan → (product, suitable_for, vegan diet)
- Extract brand relationship: (product, brand_is, brand_name)
- Extract pack size: (product, has_pack_size, "size")
- Extract product category from title: (product, is_type_of, category) — e.g. "chewing gum", "pasta", "chocolate bar"

INFERRED SEMANTIC TRIPLES (confidence 0.80) — extract ALL that apply:
- Infer USAGE OCCASION (may have MULTIPLE): (product, suitable_for_occasion, occasion) — occasions: "breakfast", "snack", "dessert", "BBQ/grilling", "lunch", "dinner", "baking ingredient", "cinema snack", "party food", "packed lunch", "picnic"
  HINTS: meats/sausages/burgers → "BBQ/grilling"; chocolate/sweets/popcorn → "cinema snack"; eggs/cereal/toast → "breakfast"; cakes/ice cream → "dessert"
- Infer COOKING METHOD (may have MULTIPLE): (product, has_cooking_method, method) — methods: "ready to eat", "microwave", "oven", "boil", "fry", "air fryer compatible", "no cooking required", "grill"
  HINTS: most meats/sausages can be grilled AND oven AND air fryer compatible; pre-cooked items are "ready to eat"
- Infer CONVENIENCE: (product, has_convenience, level) — levels: "instant", "under 5 minutes", "under 15 minutes", "requires cooking"
  HINTS: pre-cooked/sliced items → "under 5 minutes"; raw meat → "requires cooking"; instant noodles → "instant"
- Infer NUTRITIONAL PROFILE from ingredients (may have MULTIPLE): (product, has_nutritional_profile, profile) — profiles: "high protein", "high calorie", "low calorie", "high sugar", "low sugar", "high fat", "low fat", "high fiber"
  HINTS: meats/eggs/dairy → "high protein"; nuts/chocolate/peanut butter → "high calorie" + "high fat"; candy/soda → "high sugar"
- Infer PORTABILITY: (product, has_portability, level) — "pocket sized", "single serving", "sharing size", "family size", "bulk"
  HINTS: bars/small packs under 100g → "pocket sized"; 4+ servings → "family size"
- Infer TARGET AUDIENCE: (product, suitable_for_audience, audience) — "children", "adults", "athletes", "families", "health conscious"
  HINTS: high protein + portable → "athletes"; organic/sugar-free/low-fat → "health conscious"; large portions → "families"

CONFIDENCE LEVELS:
- 0.95: data directly from named fields (ingredients, allergens, claims)
- 0.85: inferred from explicit data (e.g., "contains egg" from ingredient list)
- 0.80: semantic inference (occasion, cooking method, nutritional profile from context)
- Do NOT invent information not present in the JSON — but DO make reasonable inferences from what IS there

JSON document:
{json_doc}

Return a JSON array. Each element: {{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}}

JSON array of triples:"""


# =============================================================================
# OFFLINE: Gap analysis – what triples are missing from the initial extraction
# =============================================================================

GAP_ANALYSIS_PROMPT = """You are analyzing completeness of knowledge graph extraction for food product data.

Given the original JSON document and triples already extracted, identify MISSING information.

Look for information in ANY JSON field that was not captured:
1. Individual ingredients not yet extracted (especially key ones like sugar, salt, specific proteins, eggs, milk)
2. Allergen information not captured (milk, nuts, soy, gluten, eggs, crustaceans)
3. Dietary suitability not inferred (vegan, vegetarian, halal, kosher, keto, paleo)
4. Nutritional profile not inferred from ingredients (high-protein, high-calorie, low-sugar, high-fiber)
5. Usage occasion not inferred (breakfast, snack, dessert, BBQ, baking ingredient, cinema snack, party food)
6. Cooking method/convenience not captured (ready to eat, microwave, oven, grill, air fryer compatible, boil)
7. Preparation time not inferred (instant, under 5 min, under 15 min, requires cooking)
8. Portability/format not categorized (pocket sized, single serving, sharing size, family size, bar, drink, frozen)
9. Target audience not inferred (children, athletes, families, health conscious)
10. Product category not assigned (meat, dairy, confectionery, beverage, snack, baked goods, ready meal)

Original JSON document:
{json_doc}

Already Extracted Triples:
{existing_triples}

Return a JSON array of focused extraction instructions. Each targets ONE specific gap.
Example: ["Extract all individual sweeteners from the ingredients list",
          "Infer dietary suitability (vegan/vegetarian) from ingredients"]

Gap instructions:"""


# =============================================================================
# OFFLINE: Targeted re-extraction for a specific gap
# =============================================================================

TARGETED_EXTRACTION_PROMPT = """You are a knowledge graph builder performing TARGETED extraction from a food product JSON document.

Extract ONLY triples related to this specific instruction:
>>> {gap_instruction} <<<

Do NOT duplicate triples that already exist (listed below).

Already Extracted (do NOT repeat):
{existing_triples}

JSON document:
{json_doc}

Return a JSON array. Each element: {{"subject": "...", "predicate": "...", "object": "...", "confidence": 0.9}}
Return an empty array [] if no additional triples can be extracted for this gap.

JSON array of new triples only:"""


# =============================================================================
# OFFLINE: Relationship type normalization
# =============================================================================

NORMALIZE_PREDICATES_PROMPT = """You are normalizing relationship types in a knowledge graph for food products.

Given these triples, normalize the predicate (relationship type) to a consistent vocabulary.

Use ONLY these standard predicates (pick the closest match):
- has_ingredient, has_main_ingredient
- contains_allergen, may_contain_allergen, free_from
- has_claim, has_nutritional_property
- is_type_of, is_variant_of, is_flavor_of
- brand_is, manufactured_by, country_of_origin
- has_pack_size, has_format
- suitable_for, not_suitable_for, recommended_for
- has_dietary_property (vegan, vegetarian, halal, kosher)
- has_sweetener, has_preservative, has_coloring
- has_texture, has_taste
- similar_to, alternative_to, pairs_with

If none of the standard predicates fit, keep the original but make it snake_case.

Triples to normalize:
{triples}

Return the same JSON array with predicates replaced by standard forms.

Normalized triples:"""


# =============================================================================
# OFFLINE: Entity merging decisions
# =============================================================================

ENTITY_MERGE_PROMPT = """You are resolving entity aliases in a food product knowledge graph.

Given pairs of entity names that are embedding-similar, decide if they refer to the SAME real-world entity.

For each pair, return:
- "merge": true if they are the same entity (keep the more specific/official name as canonical)
- "merge": false if they are different entities despite similar names
- "canonical": the preferred name to keep

Common merges in food: brand name variations, ingredient synonyms (e.g. "sugar" vs "sucrose"),
product name with/without brand prefix.
Do NOT merge: different flavors of same product, different sizes, related but distinct ingredients.

Pairs to evaluate:
{pairs}

Return a JSON array. Each element: {{"entity_a": "...", "entity_b": "...", "merge": true/false, "canonical": "..."}}

Decisions:"""


# =============================================================================
# ONLINE: Single-shot answer from graph context
# =============================================================================

GRAPHRAG_ANSWER_PROMPT = """You are a creative food product expert and meal planner. Use the knowledge graph and source documents below to give the BEST possible answer.

KNOWLEDGE GRAPH:
{graph_context}

SOURCE DOCUMENTS:
{source_chunks}

RULES:
1. Be helpful, creative, and specific. Recommend actual products with their product_id values.
2. Use information from the graph and source documents. You may combine multiple products into meal ideas, recipes, or usage suggestions.
3. When recommending products, explain WHY they match (ingredients, claims, nutritional properties, convenience).
4. Include product_id in bold format: **(product_id: XXXXX)**
5. ALWAYS recommend multiple products (2-4) when possible — give the user options and explain trade-offs.
6. For recipe/meal questions: suggest how to COMBINE multiple products into a complete meal. Include preparation tips and timing if available.
7. For budget questions: estimate costs and show how products work together within budget.
8. Consider dietary requirements, allergens, and claims when making recommendations.
9. If no products perfectly match, recommend the CLOSEST alternatives and explain what differs. Never say "there is no match" without offering alternatives.
10. Include relevant details: pack size, brand, key ingredients, preparation method.
11. For questions about specific ingredients (e.g. "contains eggs"): look at the ingredients lists in the source documents to find products that contain or use those ingredients.

Question: {question}

ANSWER:"""


# =============================================================================
# ONLINE: Fallback — when KG has insufficient coverage
# =============================================================================

FALLBACK_ANSWER_PROMPT = """You are a helpful food product assistant answering questions STRICTLY from the provided context.

RULES:
1. ONLY use information explicitly stated in the context below
2. Do NOT make assumptions or use external knowledge
3. Be precise and cite specific product details
4. Include product_id values for all recommended products
5. Consider dietary requirements and allergen information

Context Documents:
{context}

Question: {question}

ANSWER FROM CONTEXT:"""
