import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def get_distribution_plot(data, feature):
    # Distribution plot for a specific feature
    fig = px.histogram(data, x=feature, color='diagnosis',
                       marginal='box', nbins=30,
                       title=f"Distribution of {feature}")
    return fig


def get_box_plot(data, feature):
    # Box plot comparing Benign and Malignant cases for a specific feature
    fig = px.box(data, x='diagnosis', y=feature, color='diagnosis',
                 title=f"Box Plot of {feature} for Benign and Malignant cases")
    return fig


def get_pair_plot(data):
    # Pair plot of important features
    sns.pairplot(data[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'diagnosis']],
                 hue='diagnosis', palette='coolwarm')
    st.pyplot(plt)


def add_sidebar():
    st.header("Cell Nuclei Measurements")
    data = get_clean_data()
    
    # Group sliders by similar features: mean, se, and worst
    with st.expander("Mean Measurements", expanded=True):
        st.subheader("Mean Values")
        mean_sliders = [
            ("Radius (mean)", "radius_mean"),
            ("Texture (mean)", "texture_mean"),
            ("Perimeter (mean)", "perimeter_mean"),
            ("Area (mean)", "area_mean"),
            ("Smoothness (mean)", "smoothness_mean"),
            ("Compactness (mean)", "compactness_mean"),
            ("Concavity (mean)", "concavity_mean"),
            ("Concave points (mean)", "concave points_mean"),
            ("Symmetry (mean)", "symmetry_mean"),
            ("Fractal dimension (mean)", "fractal_dimension_mean")
        ]
        mean_inputs = {key: st.slider(label, 0.0, float(data[key].max()), float(data[key].mean())) for label, key in mean_sliders}

    with st.expander("Standard Error Measurements"):
        st.subheader("SE Values")
        se_sliders = [
            ("Radius (se)", "radius_se"),
            ("Texture (se)", "texture_se"),
            ("Perimeter (se)", "perimeter_se"),
            ("Area (se)", "area_se"),
            ("Smoothness (se)", "smoothness_se"),
            ("Compactness (se)", "compactness_se"),
            ("Concavity (se)", "concavity_se"),
            ("Concave points (se)", "concave points_se"),
            ("Symmetry (se)", "symmetry_se"),
            ("Fractal dimension (se)", "fractal_dimension_se")
        ]
        se_inputs = {key: st.slider(label, 0.0, float(data[key].max()), float(data[key].mean())) for label, key in se_sliders}

    with st.expander("Worst Measurements"):
        st.subheader("Worst Values")
        worst_sliders = [
            ("Radius (worst)", "radius_worst"),
            ("Texture (worst)", "texture_worst"),
            ("Perimeter (worst)", "perimeter_worst"),
            ("Area (worst)", "area_worst"),
            ("Smoothness (worst)", "smoothness_worst"),
            ("Compactness (worst)", "compactness_worst"),
            ("Concavity (worst)", "concavity_worst"),
            ("Concave points (worst)", "concave points_worst"),
            ("Symmetry (worst)", "symmetry_worst"),
            ("Fractal dimension (worst)", "fractal_dimension_worst")
        ]
        worst_inputs = {key: st.slider(label, 0.0, float(data[key].max()), float(data[key].mean())) for label, key in worst_sliders}

    # Merging all the inputs
    input_dict = {**mean_inputs, **se_inputs, **worst_inputs}
    return input_dict



def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}

    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value

    return scaled_dict


def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Radius', 'Texture', 'Perimeter', 'Area',
                  'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig


def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))

    input_df = pd.DataFrame([input_data], columns=input_data.keys())

    input_array_scaled = scaler.transform(input_df)

    prediction = model.predict(input_array_scaled)

    st.html("<h3>Cell cluster prediction</h3")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malicious'>Malignant</span>", unsafe_allow_html=True)

    st.write("Probability of being benign: ", model.predict_proba(input_array_scaled)[0][0])
    st.write("Probability of being malignant: ", model.predict_proba(input_array_scaled)[0][1])

    


def main():
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":female-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("Breast Cancer Predictor App")
    st.html("<p>This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.</p><p>This app predicts whether a breast mass is benign or malignant based on the measurements of cell nuclei from the tissue sample. You can also adjust the measurements using the sliders.</p>")

    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Analysis")

    col1, col2 = st.columns([4, 1])

    with col1:
        st.html("<h2> Radar Chart (Mean, Standard Error, and Worst Values)</h2><p><strong>Purpose:</strong> The radar chart visualizes the spread and variation of key features such as the radius, texture, perimeter, area, smoothness, and more, across three different measurements: mean, standard error (SE), and worst values. These features represent measurements of cell nuclei characteristics from tissue samples.</p><p><strong>Why it's useful:</strong> This chart helps in identifying the shape and distribution of these features for benign and malignant tumors. It provides a clear visual representation of how the different measurements compare. For instance, a malignant tumor may have a larger radius or more concave points, which can be observed through this chart.</p>")
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)
        

        # Display distribution plots for key features
        st.html("<h2>2. Distribution Plot (Histogram)</h2><p><strong>Purpose:</strong> The distribution plot provides a histogram for a selected feature, such as radius_mean, and color-codes it based on the diagnosis (benign or malignant). A marginal boxplot is added on the side to visualize the spread and outliers.</p><p><strong>Why it's useful:</strong> This plot helps in understanding the overall distribution of a specific feature for both benign and malignant tumors. By comparing these distributions, we can gain insights into whether certain measurements (e.g., radius_mean) are more likely to be associated with malignant cases, making it easier to differentiate between the two.</p>")
        feature_to_plot = st.selectbox("Select a feature to visualize its distribution:",
                                       options=['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean'])
        st.plotly_chart(get_distribution_plot(get_clean_data(), feature_to_plot))


        # Display box plot for feature comparison
        st.html("<h2>3. Box Plot (Feature Comparison)</h2><p><strong>Purpose:</strong> The box plot provides a side-by-side comparison of the selected feature for benign and malignant cases. The box shows the interquartile range, while the whiskers and outliers show the spread of the data.</p><p><strong>Why it's useful:</strong> Box plots are an excellent tool for understanding the range, central tendency (median), and spread of features. They also highlight outliers, which can be indicative of extreme cases. For instance, malignant tumors might have larger perimeter values compared to benign ones, and this can be easily identified through the box plot.</p>")
        st.plotly_chart(get_box_plot(get_clean_data(), feature_to_plot))

        # Display pair plot for important features
        st.html(" <h2>4. Pair Plot (Scatter Matrix)</h2><p><strong>Purpose:</strong> The pair plot visualizes pairwise relationships between key features (e.g., radius_mean, texture_mean, area_mean) along with their distributions on the diagonal.</p><p><strong>Why it's useful:</strong> This type of plot allows us to explore the relationships between pairs of features and their correlation. If features are highly correlated, they may contain similar information, which can be useful when reducing dimensionality or selecting important features. Additionally, different clusters for benign and malignant tumors may emerge, giving a clearer distinction between these two categories.</p>")
        get_pair_plot(get_clean_data())

    with col2:
        add_predictions(input_data)

    with st.sidebar:
        st.title("Contact")
        st.markdown("### Developed by:")
        st.markdown("**[Your Name]**")
        st.markdown("#### Connect with me:")
        st.markdown("[GitHub](https://github.com/yourusername)")
        st.markdown("[LinkedIn](https://www.linkedin.com/in/yourusername/)")
        st.markdown("[Twitter](https://twitter.com/yourusername)")
        st.markdown("---")
        st.markdown("**Email me at:**")
        st.markdown("<a href='mailto:youremail@example.com'>youremail@example.com</a>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
