import streamlit as st
import requests
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import shap
import pickle
#import streamlit.components.v1 as components
#import bz2
#import _pickle as cPickle

# init state
if 'btn_clicked' not in st.session_state:
    st.session_state['btn_clicked'] = False
# init state
if 'btn_clicked2' not in st.session_state:
    st.session_state['btn_clicked2'] = False

def callback1():
    # change state value
    st.session_state['btn_clicked'] = True
    st.session_state['btn_clicked2'] = False

def callback2():
    # change state value
    st.session_state['btn_clicked'] = False
    st.session_state['btn_clicked2'] = True


# Function to make API request and get prediction
@st.cache_data
def get_prediction(customer_data):
    api_url = " https://credit-scoring-sra-70fbdcfabc9e.herokuapp.com/predict"  # Replace with your API URL
    data_to_predict = {'data': customer_data.drop("SK_ID_CURR", axis=1).values.tolist()} #losrsque les modèles sans l'ID seront entrainé cela deviendra customer_data.drop("SK_ID_CURR", axis=1).values.tolist()
    response = requests.post(api_url, json=data_to_predict)

    result = response.json()
    prediction_score = result['prediction'][0]

    # Classify as 'Credit accepted' if probability of class 0 is greater than 0.5
    if prediction_score > 0.5:
        prediction_result = 'Credit accepted'
    else:
        prediction_result = 'Credit denied'
    return prediction_result, prediction_score
    
def credit_score_gauge(score):
    # Color gradient from red to yellow to green
    colors = ['#FF0000', '#FFFF00', '#00FF00']  # Red, Yellow, Green
    thresholds = [0, 0.5, 1]

    # Interpolate color based on score
    cmap = mcolors.LinearSegmentedColormap.from_list("custom", list(zip(thresholds, colors)))
    norm = mcolors.Normalize(vmin=0, vmax=1)
    #color = cmap(norm(score))

    # Plot gauge
    fig, ax = plt.subplots(figsize=(6, 0.5))  # Reduced height to accommodate lower text
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Draw color gradient as colorbar
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 0.5])

    # Draw tick marks and labels
    for i, threshold in enumerate(thresholds):
        ax.plot([threshold, threshold], [0.45, 0.5], color='black')
        ax.text(threshold, 0.55, str(threshold), fontsize=12, ha='center', va='bottom', color='black')

    # Draw dotted line at 0.5 threshold with legend
    ax.plot([0.5, 0.5], [0, 0.5], linestyle='--', color='black', label='Threshold')
    # Draw prediction indicator with legend
    ax.plot([score, score], [0, 0.5], color='black', linewidth=2, label='Client score')
    # Draw score below with the same color as the prediction indicator
    ax.text(score, -0.7, f'{score:.2f}', fontsize=14, ha='center', va='bottom', color='black')
    # Add legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.8), fancybox=True, shadow=True, ncol=2)

    st.pyplot(fig, clear_figure=True)

# Function to visualize client features
def visualize_client_features(selected_client_data, selected_feature):
    # Display position of client among others
    client_value = selected_client_data[selected_feature].values[0]
    st.text(f'Client {selected_feature}: {client_value:.2f}')

    # Plot client position in distribution
    fig, ax = plt.subplots()

    # Filter DataFrame based on prediction result
    filtered_df = df_train[df_train['TARGET'] == int(prediction_result == 'Credit denied')]

    # Check if the selected feature is categorical or continuous
    if df_train[selected_feature].dtype == 'int64':  # Categorical feature
        sns.countplot(data=filtered_df, x=selected_feature, ax=ax)
        ax.axvline(x=np.where(filtered_df[selected_feature].unique() == client_value)[0][0], color='red', linestyle='--', label=f'Client {selected_feature}')
        ax.set_title(f'Client Position in {selected_feature} Distribution')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Count')
    else:  # Continuous feature
        sns.histplot(filtered_df[selected_feature], kde=True, ax=ax)
        ax.axvline(x=client_value, color='red', linestyle='--', label=f'Client {selected_feature}')
        ax.set_title(f'Client Position in {selected_feature} Distribution')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Density')

    ax.legend()
    st.pyplot(fig, clear_figure=True)

# Function for bivariate analysis
def bivariate_analysis(feature1, feature2):
    fig, ax = plt.subplots()

    filtered_df = df_train[df_train['TARGET'] == int(prediction_result == 'Credit denied')]

    # Check the types of the selected features
    if filtered_df[feature1].dtype == 'int64' and filtered_df[feature2].dtype == 'int64':  # Categorical vs Categorical
        sns.countplot(data=filtered_df, x=feature1, hue=feature2, ax=ax)
        ax.set_title(f'Bivariate Analysis between {feature1} and {feature2}')
        ax.set_xlabel(feature1)
        ax.set_ylabel('Count')
        ax.legend(title=feature2)
    elif filtered_df[feature1].dtype != 'int64' and filtered_df[feature2].dtype != 'int64':  # Continuous vs Continuous
        sns.scatterplot(data=filtered_df, x=feature1, y=feature2, ax=ax)
        ax.set_title(f'Bivariate Analysis between {feature1} and {feature2}')
        ax.set_xlabel(feature1)
        ax.set_ylabel(feature2)
    else:  # Continuous vs Categorical or Categorical vs Continuous
        if filtered_df[feature1].dtype == 'int64':  # Categorical vs Continuous
            categorical_feature, continuous_feature = feature1, feature2
        else:  # Continuous vs Categorical
            categorical_feature, continuous_feature = feature2, feature1
        sns.boxplot(data=filtered_df, x=categorical_feature, y=continuous_feature, ax=ax)
        ax.set_title(f'Bivariate Analysis between {categorical_feature} and {continuous_feature}')
        ax.set_xlabel(categorical_feature)
        ax.set_ylabel(continuous_feature)

    st.pyplot(fig, clear_figure=True)

# Function to visualize SHAP values for the selected client
def visualize_shap_values(selected_client_data):
    st.write("Features that contribute the most to the score globally")
    # Plot global contribution
    #fig, ax = plt.subplots()
    #plt.sca(ax)
    #shap.plots.bar(shap_values, show=False, max_display=10)
    #plt.title('Global SHAP Values Analysis')
    #st.pyplot(fig)

    # Chemin vers votre image localement
    chemin_image = "utils/global_score.png"
    # Afficher l'image
    st.image(chemin_image, caption='', use_column_width=True)

    # Plot local contribution
    st.write("Features that contribute the most to the score for the selected client")
    fig, ax = plt.subplots()
    plt.sca(ax)
    shap_values_client = explainer(scaler.transform(selected_client_data.drop(columns=['SK_ID_CURR'])), max_evals=1000)
    # Plot local contribution
    force_plot_img = shap.plots.force(shap_values_client, matplotlib=True, show=False, contribution_threshold=0.07, feature_names=selected_client_data.drop(columns=['SK_ID_CURR']).columns)
    st.pyplot(force_plot_img, clear_figure=True)


##############################################################################

# Load sample parquet data
parquet_file = './data/data.parquet'
df = pq.read_table(parquet_file).to_pandas().reset_index(drop=True)

parquet_file_train = './data/train_data.parquet'
df_train = pq.read_table(parquet_file_train).to_pandas().reset_index(drop=True)

#shap_values = bz2.BZ2File('utils/shap_values.pbz2', 'rb')
#shap_values = cPickle.load(shap_values)

with open('./models/lightgbm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('./models/minmax_scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

explainer = shap.Explainer(model.predict, scaler.transform(df.drop(columns=['SK_ID_CURR'])))

#with open('utils/shap_explainer.pkl', 'rb') as f:
    # Load the data from the pickle file
    #explainer = pickle.load(f)

# Streamlit app with sidebar
st.sidebar.title('Credit Scoring Prediction Dashboard')

# Dropdown for client IDs in the sidebar
selected_client_id = st.sidebar.selectbox('Select Client ID:', df['SK_ID_CURR'].unique())

# Display selected client's data in the main section
st.sidebar.header('Selected Client Data:')
selected_client_data = df.loc[df['SK_ID_CURR'] == selected_client_id]
st.sidebar.write(selected_client_data)



# Button to trigger prediction in the sidebar
if st.sidebar.button('Predict', on_click=callback1) or st.session_state['btn_clicked']:
    # Make API request and get prediction
    prediction_result, prediction_score = get_prediction(selected_client_data)

    # Display prediction result
    st.sidebar.subheader('Prediction Result:')
    if prediction_result is not None:
        # Determine emoji based on prediction result
        emoji = "❌" if prediction_result == "Credit denied" else "✅"

        # Display prediction result with emoji
        st.sidebar.write(f"{emoji} The credit is accepted if the score is greater than 0.5 or 50%, denied otherwise. In this case, the predicted score is {prediction_score:.2}")

        st.sidebar.write(f"{emoji} The credit status is: {prediction_result}")
        st.sidebar.write(f"{emoji} The prediction score is: {prediction_score:.2%}")
        st.sidebar.write(f"{emoji} The probability is: {prediction_score:.2}")


        # Visualisation du score de crédit (jauge colorée)
        st.subheader('Credit Score Visualization:')
        credit_score_gauge(prediction_score)
        st.text("A color gauge representing the credit score. The client's score is indicated by a marker on the gauge.")

        # Visualisation de la contribution des features
        st.subheader('Feature Contribution:')
        visualize_shap_values(selected_client_data)
        st.text("Bar chart and force plot showing the features that contribute the most to the credit score globally and for the selected client.")

        # Dropdown for feature selection
        selected_feature = st.selectbox('Select Feature:', df.drop(columns=['SK_ID_CURR']).columns, key='feature_selection')
        st.text("A graphical representation of the client's position among others based on the selected feature for the same target as the client.")

        # Display client features visualization
        visualize_client_features(selected_client_data, selected_feature)

        # Graphique d’analyse bi-variée entre deux features sélectionnées
        st.subheader('Bi-variate Analysis:')
        # Select two features for bivariate analysis
        selected_feature1 = st.selectbox('Select Feature 1:', df.drop(columns=['SK_ID_CURR']).columns, key='feature_selection1')
        selected_feature2 = st.selectbox('Select Feature 2:',df.drop(columns=['SK_ID_CURR']).columns, key='feature_selection2')

        # Display bivariate analysis
        bivariate_analysis(selected_feature1, selected_feature2)
        st.text("A graphical analysis of the relationship between two selected features for the same target as the client.")

# Allow user to modify client information
st.sidebar.subheader('Modify Client Information:')

# Get columns to modify
columns_to_modify = st.sidebar.multiselect('Select columns to modify:', selected_client_data.columns)

# Dictionary to store updates
client_data_updates = {}

# Loop through selected columns
for col in columns_to_modify:
    new_value = st.sidebar.text_input(f'Enter new value for {col}:', value=selected_client_data[col].iloc[0])

    # Check entered values
    if selected_client_data[col].dtype == 'int64':  # If the column is categorical
        new_value = int(new_value)
        unique_values = df_train[col].unique()
        if new_value not in unique_values:
            st.warning(f"The value must be among: {', '.join(map(str, unique_values))}")
            continue
    else:  # If the column is numerical
        min_value = df_train[col].min()
        max_value = df_train[col].max()
        try:
            new_value = float(new_value)
            if new_value < min_value or new_value > max_value:
                st.warning(f"The value must be between {min_value} and {max_value}")
                continue
        except ValueError:
            st.warning("The value must be a valid number for a continuous variable.")
            continue

    client_data_updates[col] = new_value

# Button to trigger prediction with updated information
if st.sidebar.button('Update and Re-predict', on_click=callback2) or st.session_state['btn_clicked2']:
    # Update client data
    for col, new_value in client_data_updates.items():
        selected_client_data.loc[:, col] = new_value

    # Make API request and get updated prediction
    prediction_result, prediction_score = get_prediction(selected_client_data)

    # Display updated prediction result
    st.sidebar.subheader('Updated Prediction Result:')
    # Display prediction result for new client
    if prediction_result is not None:
        # Determine emoji based on prediction result
        emoji = "❌" if prediction_result == "Credit denied" else "✅"

        # Display prediction result with emoji
        st.sidebar.write(f"{emoji} The credit is accepted if the score is greater than 0.5 or 50%, denied otherwise. In this case, the predicted score is {prediction_score:.2}")

        st.sidebar.write(f"{emoji} The credit status is: {prediction_result}")
        st.sidebar.write(f"{emoji} The prediction score is: {prediction_score:.2%}")
        st.sidebar.write(f"{emoji} The probability is: {prediction_score:.2}")


        # Visualisation du score de crédit (jauge colorée)
        st.subheader('Credit Score Visualization:')
        credit_score_gauge(prediction_score)
        st.text("A color gauge representing the credit score. The client's score is indicated by a marker on the gauge.")

        # Visualisation de la contribution des features
        st.subheader('Feature Contribution:')
        visualize_shap_values(selected_client_data)
        st.text("Bar chart and force plot showing the features that contribute the most to the credit score globally and for the selected client.")

        st.subheader('Client features visualization:')
        # Dropdown for feature selection
        selected_feature = st.selectbox('Select Feature:', df.drop(columns=['SK_ID_CURR']).columns, key='feature_selection_mod')
        st.text("A graphical representation of the client's position among others based on the selected feature for the same target as the client.")

        # Display client features visualization
        visualize_client_features(selected_client_data, selected_feature)

        # Graphique d’analyse bi-variée entre deux features sélectionnées
        st.subheader('Bi-variate Analysis:')
        # Select two features for bivariate analysis
        selected_feature1 = st.selectbox('Select Feature 1:', df.drop(columns=['SK_ID_CURR']).columns, key='feature_selection1_mod')
        selected_feature2 = st.selectbox('Select Feature 2:',df.drop(columns=['SK_ID_CURR']).columns, key='feature_selection2_mod')

        # Display bivariate analysis
        bivariate_analysis(selected_feature1, selected_feature2)
        st.text("A graphical analysis of the relationship between two selected features for the same target as the client.")