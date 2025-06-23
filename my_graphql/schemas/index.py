# my_graphql/schemas/index.py

type_defs = """
type Query {
  getSemanticSimilarities(path: String!, ranking:String!, threshold: Float!, k: Int!): [SemanticSimilarities]
  getAPIIntegrationPoints_Semantic(apiList: [inputAPI], ranking: String!, threshold: Float!, k: Int!): APIIntegrationPoints_Semantic
}
input inputAPI {
		name: String
		path: String
}
type SemanticSimilarities {
  origin_api: String,
  target_api: String,
  oa_out_attr: String,
  ta_in_attr: String,
  oa_out_attr_dt: String,
  ta_in_attr_dt:String,
  oa_out_attr_parent: String,
  ta_in_attr_parent: String,
  path: Float,
  lch: Float,
  wup: Float,
  hso: Float,
  lin_x: Float,
  tversky: Float,
  lesk: Float,
  resnik: Float,
  jcn: Float,
  lin_y: Float,
  word2vec: Float,
  glove: Float,
  fasttext: Float,
  bert: Float,
  lstm: Float,
  bilstm : Float,
  roberta: Float,
  mpnet: Float, 
  major_rank_sem: Float, 
  gini_sem: Float, 
  avg_rank_sem: Float, 
  prediction_sem: Float
}

type APIIntegrationPoints_Semantic {
		generatedExtractedFile: String
		generatedDatasetFile: String
    semantic:[SemanticSimilarities]
}
"""
