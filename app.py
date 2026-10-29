import gradio as gr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train_data = pd.read_csv('train.csv')
features = ["Pclass", "Sex", "SibSp", "Parch"]

X = pd.get_dummies(train_data[features])
y = train_data["Survived"]

model = RandomForestClassifier(n_estimators= 100, max_depth = 5, random_state = 42)
model.fit(X, y)

def predict_survival(Pclass, Sex, SibSp, Parch):
    input_data = pd.DataFrame({
        'Pclass': [int(Pclass)],
        'SibSp': [int(SibSp)],
        'Parch': [int(Parch)],
        'Sex_female': [1 if Sex == 'female' else 0],
        'Sex_male': [1 if Sex == 'male' else 0]
    })

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        return f"Sobreviveria! (Probabilidade: {prob:.1f})"
    else:
        return f"Não sobreviveria! (Probabilidade: {prob:.1f})"


demo = gr.Interface(
    fn = predict_survival,
    inputs = [
        gr.Radio(["1", "2", "3"], label = "CLasse do bilhete (Pclass)"),
        gr.Radio(["male", "female"], label = "Sexo"),
        gr.Slider(0, 8, step = 1, label = "Irmãos/Cônjuges a bordo (SibSp)"),
        gr.Slider(0, 6, step = 1, label = "Pais/Filhos a bordo (Parch)")
    ],

    outputs = "text",
    title = "Previsão de Sobrevivência no Titanic",
    description = "Modelo de Random Forest treinado para prever a sobrevivência de passageiros do Titanic com base em suas características."
)

demo.launch()