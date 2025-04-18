import streamlit as st
import numpy as np
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import joblib
import os
import hashlib
import pandas as pd

st.set_page_config(page_title="Additionneur IA", layout="centered")
st.title("Additionneur IA (réseau neuronal)")
st.markdown("Un petit réseau neuronal apprend à faire la somme de deux nombres entre 0 et 100.")

if "history" not in st.session_state:
    st.session_state.history = []

st.subheader("Qualité des données")
data_quality = st.selectbox("Choisissez la qualité des données :", ["Excellente (sans erreur)", "Moyenne (5% bruit)", "Faible (20% bruit)"])

st.subheader("Structure du réseau neuronal")
nb_layers = st.slider("Nombre de couches cachées", 1, 3, 1)
neurons_per_layer = st.slider("Neurones par couche", 1, 20, 5)
hidden_layers = tuple([neurons_per_layer] * nb_layers)
st.markdown(f"Structure choisie : {hidden_layers}")

@st.cache_resource
def train_model(hidden_layers, data_quality):
    X = np.random.randint(0, 100, (1000, 2))
    y = np.sum(X, axis=1)

    if data_quality == "Moyenne (5% bruit)":
        indices = np.random.choice(1000, size=50, replace=False)
        y[indices] += np.random.randint(-10, 10, size=50)
    elif data_quality == "Faible (20% bruit)":
        indices = np.random.choice(1000, size=200, replace=False)
        y[indices] += np.random.randint(-20, 20, size=200)

    X_norm = X / 100
    y_norm = y / 200

    model = MLPRegressor(hidden_layer_sizes=hidden_layers, activation='relu', solver='adam',
                         learning_rate_init=0.01, max_iter=300)
    model.fit(X_norm, y_norm)
    return model

model = train_model(hidden_layers, data_quality)

# Évaluation sur 10 exemples
X_test = np.random.randint(0, 100, (10, 2))
y_test = np.sum(X_test, axis=1)
X_test_norm = X_test / 100
y_test_norm = y_test / 200
y_pred_norm = model.predict(X_test_norm)
y_pred = y_pred_norm * 200
mse = np.mean((y_pred - y_test)**2)

# Sauvegarder l'historique
st.session_state.history.append({
    "Qualité des données": data_quality,
    "Structure": str(hidden_layers),
    "Perte finale": model.loss_,
    "Erreur test MSE": round(mse, 2)
})

# Sauvegarde du modèle
config_str = f"{data_quality}_{hidden_layers}"
config_hash = hashlib.md5(config_str.encode()).hexdigest()
filename = f"streamlit_additionneur/models/model_{config_hash}.joblib"
os.makedirs("streamlit_additionneur/models", exist_ok=True)
joblib.dump(model, filename)
st.caption(f"Modèle sauvegardé dans : `{filename}`")

# Interface test
st.subheader("Tester le modèle")
mode_test = st.radio("Mode de saisie :", ["Sliders", "Clavier"])
if mode_test == "Sliders":
    a = st.slider("Nombre 1", 0, 99, 42)
    b = st.slider("Nombre 2", 0, 99, 27)
else:
    a = st.number_input("Nombre 1", 0, 99, 42)
    b = st.number_input("Nombre 2", 0, 99, 27)

x = np.array([[a / 100, b / 100]])
pred = model.predict(x)[0] * 200
somme = a + b

st.markdown(f"### Prédiction IA : **{pred:.2f}**")
st.markdown(f"### Somme exacte : **{somme}**")
diff = abs(pred - somme)
if diff <= 1:
    st.success("Excellente prédiction !")
elif diff <= 5:
    st.warning("Prédiction acceptable.")
else:
    st.error("Prédiction mauvaise. Le réseau a mal appris.")

# Résultats enregistrés
st.subheader("Résultats enregistrés")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    if len(df) >= 2:
        fig, ax = plt.subplots()
        for qual in df["Qualité des données"].unique():
            subset = df[df["Qualité des données"] == qual]
            ax.plot(subset["Structure"], subset["Erreur test MSE"], label=qual, marker='o')
        ax.set_ylabel("Erreur MSE")
        ax.set_xlabel("Structure")
        ax.legend()
        st.pyplot(fig)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger en CSV", data=csv, file_name="resultats_models.csv", mime="text/csv")

if st.button("Réinitialiser les résultats"):
    st.session_state.history = []
    st.success("Résultats effacés.")