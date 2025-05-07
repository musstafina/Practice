import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack, csr_matrix

food_df = pd.read_csv("FINAL_COMBINED.csv")
disease_df = pd.read_csv("Disease.csv")

selected_nutrients = [
    'protein', 'fat', 'carbohydrate (nfe)', 'crude fibre', 'calcium',
    'phospohorus', 'potassium', 'sodium', 'magnesium', 'vitamin e',
    'vitamin c', 'omega-3-fatty acids', 'omega-6-fatty acids'
]

for col in selected_nutrients:
    food_df[col] = food_df[col].astype(str).str.replace('%', '').str.replace('IU/kg', '').str.extract(r'([\d.]+)').astype(float)


food_df['combined_text'] = (
    food_df['product title'].fillna('') + ' ' +
    food_df['product description'].fillna('') + ' ' +
    food_df['key benefits'].fillna('') + ' ' +
    food_df['ingredients'].fillna('') + ' ' +
    food_df['helpful tips'].fillna('') + ' ' +
    food_df['need/preference'].fillna('') + ' ' +
    food_df['alternate product recommendation'].fillna('') 
)
# TF-IDF + SVD
vectorizer = TfidfVectorizer(stop_words='english')
X_text = vectorizer.fit_transform(food_df['combined_text'])
svd = TruncatedSVD(n_components=300, random_state=42)
X_text_reduced = svd.fit_transform(X_text)

encoder = OneHotEncoder(sparse_output=True)
X_categorical = encoder.fit_transform(food_df[['breed size', 'lifestage', 'food form']].fillna('Unknown'))
X_combined = hstack([csr_matrix(X_text_reduced), X_categorical])

# scalers = {}
# scaled_targets = {}
# for nutrient in selected_nutrients:
#     y = food_df[nutrient].fillna(food_df[nutrient].median())
#     if nutrient in ['sodium', 'omega-3-fatty acids', 'omega-6-fatty acids']:
#         scaler = StandardScaler()
#         y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
#         scalers[nutrient] = scaler
#         scaled_targets[nutrient] = y_scaled
#     else:
#         scaled_targets[nutrient] = y

scale_nutrients = [
    'sodium', 'omega-3-fatty acids', 'omega-6-fatty acids',
    'calcium', 'phospohorus', 'potassium', 'magnesium'
]

scalers = {}
scaled_targets = {}

for nutrient in selected_nutrients:
    y = food_df[nutrient].fillna(food_df[nutrient].median())
    if nutrient in scale_nutrients:
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
        scalers[nutrient] = scaler
        scaled_targets[nutrient] = y_scaled
    else:
        scaled_targets[nutrient] = y

def train_ridge(X, y):
    ridge = Ridge()
    grid = GridSearchCV(ridge, {'alpha': [0.1, 1.0, 10.0]}, scoring='r2', cv=3)
    grid.fit(X, y)
    return grid.best_estimator_

ridge_models = {}
for nutrient in selected_nutrients:
    y = scaled_targets[nutrient]
    X_train, _, y_train, _ = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    ridge_models[nutrient] = train_ridge(X_train, y_train)

disorder_keywords = {
    "Inherited musculoskeletal disorders": "joint mobility glucosamine arthritis cartilage flexibility osteoarthritis chondroprotective msm collagen anti-inflammatory",
    "Inherited gastrointestinal disorders": "digest stomach bowel sensitive diarrhea gut ibs prebiotic probiotic microbiome fiber motility",
    "Inherited endocrine disorders": "thyroid metabolism weight diabetes insulin hormone glucose glycemic hypothyroidism hyperthyroidism",
    "Inherited eye disorders": "vision eye retina cataract antioxidant lutein zeaxanthin beta-carotene ocular macular degeneration",
    "Inherited nervous system disorders": "brain seizure cognitive nerve neuro neurological cognition dementia epilepsy dha epa omega-3 neuroprotective",
    "Inherited cardiovascular disorders": "heart cardiac circulation omega-3 blood pressure vascular taurine l-carnitine coenzyme q10 arrhythmia hypertension",
    "Inherited skin disorders": "skin allergy itch coat omega-6 dermatitis eczema flaky seborrhea zinc fatty acids biotin histamine",
    "Inherited immune disorders": "immune defense resistance inflammatory autoimmune antioxidant vitamin e vitamin c cytokines immunomodulatory",
    "Inherited urinary and reproductive disorders": "urinary bladder kidney renal urine reproductive struvite crystals oxalate ph urinary tract infection",
    "Inherited respiratory disorders": "breath respiratory airway lung cough breathing nasal bronchodilator antihistamine pulmonary inflammation",
    "Inherited blood disorders": "anemia blood iron hemoglobin platelets clotting hemophilia erythropoiesis vitamin b12 folate"
}


st.title("Dog Breed Nutrition Recommendation")

breeds = disease_df['Breed'].unique()
selected_breed = st.selectbox("Select Dog Breed", breeds)

breed_disorders = disease_df[disease_df['Breed'] == selected_breed]['Disease'].unique()
selected_disorder = st.selectbox("Select Disorder", breed_disorders)

if st.button("Generate Recommendation"):
    disorder_type = disease_df[(disease_df['Breed'] == selected_breed) & 
                               (disease_df['Disease'] == selected_disorder)]['Disorder'].values[0]
    keyword_string = disorder_keywords.get(disorder_type, '')
    keyword_vec = vectorizer.transform([keyword_string])
    keyword_reduced = svd.transform(keyword_vec)
    keyword_combined = hstack([csr_matrix(keyword_reduced), encoder.transform([['Unknown', 'Unknown', 'Unknown']])])

    # Recommend recipe
    similarities = cosine_similarity(keyword_vec, vectorizer.transform(food_df['combined_text'])).flatten()
    top_idx = similarities.argmax()
    recommended_product = food_df.iloc[top_idx]['product title']

    st.subheader(f"Recommended Recipe: {recommended_product}")

    # Nutrient forecasting
    forecast = {}
    for nutrient, model in ridge_models.items():
        pred = model.predict(keyword_combined)[0]
        if nutrient in scalers:
            pred = scalers[nutrient].inverse_transform([[pred]])[0][0]
        forecast[nutrient] = round(pred, 2)

    forecast_df = pd.DataFrame(list(forecast.items()), columns=["Nutrient", "Forecasted Value"])
    st.dataframe(forecast_df)
