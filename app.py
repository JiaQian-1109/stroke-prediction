import streamlit as st
import joblib
import numpy as np
import base64  # Import the base64 module
import pickle

# Background Image CSS
def set_background(local_image_path):
    with open(local_image_path, "rb") as f:
        image_data = f.read()
    # Encode the image data to base64
    base64_image = base64.b64encode(image_data).decode("utf-8")

    page_bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{base64_image}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    

    .centered-button {{
            display: flex;
            justify-content: center;
            
        }}
    div.stButton > button:first-child {{
        background-color: #81d8d2;
        color: white;
        padding: 15px 80px;
        font-size: 25px;
        border-radius: 10px;
        border: none;
        cursor: pointer;
    }}
    div.stButton > button:first-child:hover {{
        background-color: #88e9e3;
        color: white;
    }}

    .stButton>button {{
    display: block;
    margin: 0 auto;
    }}

    label, div[data-testid="stMarkdown"] p {{
        color: white !important;
        font-size: 18px;
    }}

    
    
    

    </style>
    """
    st.markdown(page_bg_css, unsafe_allow_html=True)

# Set Background
set_background("mainpage.png")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"

# Sidebar Navigation
st.sidebar.title("Navigation")
if st.sidebar.button("Home"):
    st.session_state.page = "Home"
    st.rerun()
if st.sidebar.button("About Us"):
    st.session_state.page = "About Us"
    st.rerun()

# Page Routing
if st.session_state.page == "Home":
    # Custom CSS for specific white text
    

    # Wrap the text in a div with the class "white-text"
    
    st.markdown(
        """
        <div style="text-align: center; line-height: 1.8;">
            <h1 style="color: white; font-size: 80px; margin-bottom: 10px; margin-left: 40px; margin-top:100px; ">Stroke Detection</h1>
            <p style="color: white; font-size: 25px; margin-bottom: 5px;">Your Body is Your Home</p>
            <p style="color: white; font-size: 25px; margin-bottom: 80px;">Build Your Health To Make It Stronger</p>
        </div>

        <style>
        
        </style>
        """,
        unsafe_allow_html=True
    )
    
    if st.button("Start"):
        st.session_state.page = "Selection"
        st.rerun()

elif st.session_state.page == "Selection":
    

    st.markdown(
        """
        <div style="text-align: center; line-height: 1.8;">
            <h1 style="color: white; font-size: 60px; margin-bottom: 10px; margin-left: 40px; margin-top:100px; ">Selection Page</h1>
            <p style="color: white; font-size: 25px; margin-bottom: 5px;">Choose an option:</p>
            
        </div>

       
        """,
        unsafe_allow_html=True
    )
    
    if st.button("Stroke Detection"):
        st.session_state.page = "Stroke Detection"
        st.rerun()

    if st.button("View Model Performance"):
        st.session_state.page = "Model Performance"
        st.rerun()

elif st.session_state.page == "Stroke Detection":
    st.markdown(
        """
        <div style="text-align: center; line-height: 1.8;">
            <h1 style="color: white; font-size: 60px; margin-bottom: 10px; margin-left: 40px; margin-top:100px; ">Stroke Detection</h1>
            <p style="color: white; font-size: 25px; margin-bottom: 5px; text-align: left; font-weight: bold;">1. Please fill your information</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Check for saved input data
    saved_values = st.session_state.get("saved_input", {})

    # Input fields
    # Input fields with saved values
    age = st.number_input("How old is the patient?", min_value=0, max_value=120, value=saved_values.get("age", 30))
    hypertension = st.selectbox("Does the patient have a history of hypertension?", ["No", "Yes"], index=saved_values.get("hypertension", 0))
    heart_disease = st.selectbox("Does the patient have a history of heart disease?", ["No", "Yes"], index=saved_values.get("heart_disease", 0))
    avg_glucose_level = st.number_input("What is the patient's average glucose level?", min_value=0.0, value=saved_values.get("avg_glucose_level", 100.0))
    bmi = st.number_input("What is the patient's Body Mass Index (BMI)?", min_value=0.0, value=saved_values.get("bmi", 25.0))
    smoking_status = st.selectbox("What is the patient's smoking status?", ["Never smoked", "Formerly smoked", "Smokes", "Unknown"], index=saved_values.get("smoking_status", 0))
    work_type = st.selectbox("What is the patient's occupation type?", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"], index=saved_values.get("work_type", 0))
    ever_married = st.selectbox("Has the patient ever been married?", ["No", "Yes"], index=saved_values.get("ever_married", 0))

    # Encode categorical variables
    hypertension_encoded = 1 if hypertension == "Yes" else 0
    heart_disease_encoded = 1 if heart_disease == "Yes" else 0
    ever_married_encoded = 1 if ever_married == "Yes" else 0
    work_type_map = {"Children": 0, "Govt_job": 1, "Never_worked": 2, "Private": 3, "Self-employed": 4}
    smoking_map = {"Formerly smoked": 0, "Never smoked": 1, "Smokes": 2, "Unknown": 3}

    work_type_encoded = work_type_map[work_type]
    smoking_status_encoded = smoking_map[smoking_status]

    # Model Selection
    st.markdown(
        """
        <div style="text-align: center; line-height: 1.8;">
            <p style="color: white; font-size: 25px; margin-bottom: 5px; text-align: left; font-weight: bold;">2. Please select the model</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    model_choice = st.selectbox(
        "Choose a model:", 
        ["SVM + RF", "SVM + ANN"],
        index=0 if st.session_state.get('saved_input', {}).get('model_choice', 'SVM + RF') == "SVM + RF" else 1
    )


    # Loading (in your Streamlit app):
    with open("ann_model.pkl", "rb") as f:
        ann_model = pickle.load(f)

    # Load models based on selection
    if model_choice == "SVM + RF":
        try:
            rf_model = joblib.load("rf_model_best.pkl")
            svm_model = joblib.load("hybrid_model_rf_svm.pkl") #hybrid_model_rf.pkl

            # Ensure model files are loaded properly
            if not rf_model or not svm_model:
                raise ValueError("One or more models failed to load.")
        
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
    else:  # SVM + ANN
        try:
            # Load all components using joblib
            svm_model = joblib.load("svm_model.pkl")
            ann_model = joblib.load("ann_model.pkl")  # Assuming you've saved ANN with joblib
            meta_classifier = joblib.load("meta_classifier.pkl")
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")

    # Predict Button
    if st.button("Predict Stroke Risk"):
        # Convert input to NumPy array
        input_data = np.array([[age, hypertension_encoded, heart_disease_encoded, 
                               avg_glucose_level, bmi, smoking_status_encoded, work_type_encoded, ever_married_encoded]])
        
        try:
            # Process based on model choice
            if model_choice == "SVM + RF":
                # First get RF predictions
                rf_prob = rf_model.predict_proba(input_data)[:, 1]
                rf_class = rf_model.predict(input_data)
                # Combine original features with RF outputs
                augmented_data = np.column_stack([input_data, rf_prob, rf_class])
                # Make final prediction with SVM
                prediction = svm_model.predict(augmented_data)
            else:  # SVM + ANN
                # Get predictions from base models
                svm_probs = svm_model.predict_proba(input_data)[:, 1]
                ann_probs = ann_model.predict(input_data)[:, 0]  # Using predict_proba
                
                # Combine predictions
                combined = np.column_stack((svm_probs, ann_probs))
                
                # Final prediction
                prediction = meta_classifier.predict(combined)
            
            # Store the results in session state
            st.session_state.result_data = {
                "Age": age,
                "Hypertension": hypertension,
                "Heart Disease": heart_disease,
                "Average Glucose Level": avg_glucose_level,
                "BMI": bmi,
                "Smoking Status": smoking_status,
                "Work Type": work_type,
                "Ever Married": ever_married,
                "Prediction Result": "High Risk of Stroke" if prediction[0] == 1 else "Low Risk of Stroke"
            }
            st.session_state.model_choice = model_choice
          
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.stop()  # Stop execution if there's an error
            
        # Navigate to Result page
        st.session_state.page = "Result"
        st.rerun()

    # Exit Button
    if st.button("Exit to Home"):
        st.session_state.page = "Home"
        st.rerun()

elif st.session_state.page == "Result":
   
    st.markdown(
    """
    <div style="text-align: center; line-height: 1.8;">
        <h1 style="color: white; font-size: 60px; margin-bottom: 10px; margin-left: 40px; margin-top:100px; ">Stroke Detection Summary</h1>
        <p style="color: white; font-size: 25px; margin-bottom: 5px; font-weight: bold;">Here is a summary of your inputted data and the prediction result:</p>
        
    </div>
    """,
    unsafe_allow_html=True
    )

    # Check if result_data exists in session_state
    if "result_data" in st.session_state:
        result_data = st.session_state.result_data

        # Use HTML/CSS for tidy formatting
        st.markdown(
            f"""
            <style>
                .summary-container {{
                    margin-bottom: 10px;
                }}
                .question {{
                    display: inline-block;
                    width: 400px;
                    color: white;
                }}
                .answer {{
                    font-weight: bold;
                    display: inline-block;
                    color: white;
                }}
            </style>
            <div class="summary-container">
                <div class="question">How old is the patient?</div>
                <div class="answer"><strong>{result_data['Age']} years</strong></div>
            </div>
            
            <div class="summary-container">
                <div class="question">Does the patient have a history of hypertension?</div>
                <div class="answer"><strong>{result_data['Hypertension']}</strong></div>
            </div>
            <div class="summary-container">
                <div class="question">Does the patient have a history of heart disease?</div>
                <div class="answer"><strong>{result_data['Heart Disease']}</strong></div>
            </div>
            
            <div class="summary-container">
                <div class="question">What is the patient's average glucose level?</div>
                <div class="answer"><strong>{result_data['Average Glucose Level']}</strong></div>
            </div>
            <div class="summary-container">
                <div class="question">What is the patient's Body Mass Index (BMI)?</div>
                <div class="answer"><strong>{result_data['BMI']}</strong></div>
            </div>
            <div class="summary-container">
                <div class="question">What is the patient's smoking status?</div>
                <div class="answer"><strong>{result_data['Smoking Status']}</strong></div>
            </div>
            <div class="summary-container">
                <div class="question">What is the patient's occupation type?</div>
                <div class="answer"><strong>{result_data['Work Type']}</strong></div>
            </div>
            <div class="summary-container">
                <div class="question">Has the patient ever been married?</div>
                <div class="answer"><strong>{result_data['Ever Married']}</strong></div>
            </div>
                <div class="summary-container">
                <div class="question">Model Used:</div>
                <div class="answer"><strong>{st.session_state.get('model_choice', 'Not specified')}</strong></div>
            </div>

            """,
            unsafe_allow_html=True
        )
        #model_choice
        # Display the prediction result
        st.markdown(
        """
        <div style="text-align: center; line-height: 1.8;">
            <p style="color: white; font-size: 25px; margin-bottom: 5px; text-align: left; font-weight: bold; ">Prediction Result: </p>
            
        </div>

        
        """,
        unsafe_allow_html=True
        )
        if result_data['Prediction Result'] == "High Risk of Stroke":
            st.markdown(
                f"""
                <p style="color: red; font-size: 22px; font-weight: bold; background-color: #d26060;">
                    ⚠️ {result_data['Prediction Result']}! Please consult a doctor.
                </p>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <p style="color: green; font-size: 22px; font-weight: bold; background-color: #69da87;">
                    ✅ {result_data['Prediction Result']}. Keep maintaining a healthy lifestyle!
                </p>
                """,
                unsafe_allow_html=True
            )
        # Prepare the content for download
        download_content = (
            "Stroke Detection Summary\n\n"
            "Input Summary:\n"
            f"- How old is the patient?                           {result_data['Age']} years old\n"
            f"- Does the patient have a history of hypertension?  {result_data['Hypertension']}\n"
            f"- Does the patient have a history of heart disease? {result_data['Heart Disease']}\n"
            f"- What is the patient's average glucose level?      {result_data['Average Glucose Level']}\n"
            f"- What is the patient's Body Mass Index (BMI)?      {result_data['BMI']}\n"
            f"- What is the patient's smoking status?             {result_data['Smoking Status']}\n"
            f"- What is the patient's occupation type?            {result_data['Work Type']}\n"
            f"- Has the patient ever been married?                {result_data['Ever Married']}\n\n"
            "Prediction Result:\n"
            f"{result_data['Prediction Result']}"
        )

        # Download Button
        st.download_button(
            label="Download Summary",
            data=download_content,
            file_name="stroke_detection_summary.txt",
            mime="text/plain"
        )

        # Back to Stroke Detection or Home Button
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Back to Stroke Detection"):
                # Store the current input data in session_state before navigating back
                if "result_data" in st.session_state:
                    # Convert Yes/No to index values (0 for No, 1 for Yes)
                    hypertension_index = 1 if st.session_state.result_data["Hypertension"] == "Yes" else 0
                    heart_disease_index = 1 if st.session_state.result_data["Heart Disease"] == "Yes" else 0
                    ever_married_index = 1 if st.session_state.result_data["Ever Married"] == "Yes" else 0
                    
                    # Get smoking status index
                    smoking_options = ["Never smoked", "Formerly smoked", "Smokes", "Unknown"]
                    smoking_index = smoking_options.index(st.session_state.result_data["Smoking Status"]) if st.session_state.result_data["Smoking Status"] in smoking_options else 0
                    
                    # Get work type index
                    work_options = ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"]
                    work_index = work_options.index(st.session_state.result_data["Work Type"]) if st.session_state.result_data["Work Type"] in work_options else 0
                    
                    st.session_state.saved_input = {
                        "age": st.session_state.result_data["Age"],
                        "hypertension": hypertension_index,
                        "heart_disease": heart_disease_index,
                        "avg_glucose_level": st.session_state.result_data["Average Glucose Level"],
                        "bmi": st.session_state.result_data["BMI"],
                        "smoking_status": smoking_index,
                        "work_type": work_index,
                        "ever_married": ever_married_index,
                        "model_choice": st.session_state.model_choice
                    }
                st.session_state.page = "Stroke Detection"
                st.rerun()
                
        with col2:
            if st.button("Exit to Home"):
                st.session_state.page = "Home"
                st.rerun()
              
    else:
        st.error("No data available. Please complete the Stroke Detection form first.")
        if st.button("Go to Stroke Detection"):
            st.session_state.page = "Stroke Detection"
            st.rerun()


     
    st.markdown("""
    <div class="warning-text">
        <br><br><br>
        <p>***The rusult might not 100% accurate due to the imbalance dataset.***</p>
                
    </div>
    """, unsafe_allow_html=True)

   

elif st.session_state.page == "Model Performance":

    st.markdown(
    """
    <style>
    .white-title {
        color: white;
        font-size: 2.5em;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
    st.markdown(
        '<div class="white-title">Model Performance</div>'
        , unsafe_allow_html=True)
    # Model Selection
    st.write("Please select the model")
    model_choice = st.selectbox("Choose a model:", ["SVM + RF", "SVM + ANN"])

    # Load the corresponding model
    if model_choice == "SVM + RF":
        with open("svmrf_model_performance.txt", "r") as f:
            performance = f.read()
    else:  # SVM + ANN
        with open("svmann_model_performance.txt", "r") as f:
            performance = f.read()

    
    st.write("View model performance metrics here.")
    # Split the performance metrics into lines
    metrics = performance.strip().split("\n")

    # Create two columns for metric names and values
    col1, col2 = st.columns(2)

    # Display metric names in the first column and values in the second column
    col1.write("**Metric Name**")
    col2.write("**Metric Value**")
    
    for metric in metrics:
        if ": " in metric:  # Ensure the line contains a metric name and value
            name, value = metric.split(": ")
            col1.write(name)
            col2.write(value)
    if st.button("Back to Home"):
        st.session_state.page = "Home"
        st.rerun()

elif st.session_state.page == "About Us":
    st.markdown("""
    <style>
        .about-us-text {
            color: white !important;
            font-size: 18px;
            line-height: 1.6;
        }
        .about-us-title {
            color: white !important;
            font-size: 36px !important;
            text-align: center;
            margin-bottom: 30px;
        }
        .about-us-list {
            color: white !important;
            font-size: 18px;
            margin-left: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="about-us-title">About Us</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-us-text">
        Welcome to Build Your Health, a dedicated initiative focused on advancing stroke prediction and prevention. 
        Our mission is to empower individuals and healthcare providers with cutting-edge tools and insights to 
        identify and mitigate the risk of stroke. We combine advanced technology, medical expertise, 
        and data-driven analytics to create personalized risk assessments.
        <br><br>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-us-text">
        By understanding your unique health profile, we aim to guide you toward proactive measures that could save lives.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-us-text">
        Our approach integrates:
    </div>
    <ul class="about-us-list">
        <li>Predictive algorithms backed by the latest medical research.</li>
        <li>Comprehensive analysis of individual health metrics.</li>
        <li>Education and resources to help you make informed lifestyle choices.</li>
    </ul>
    <br>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-us-text">
        At Build Your Health, we believe that awareness is the first step in prevention. 
        Together, we can take action to build healthier futures.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-us-text">
        Contributor:
                <br><br><br>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Back to Home"):
        st.session_state.page = "Home"
        st.rerun()


