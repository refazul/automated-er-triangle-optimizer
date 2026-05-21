import streamlit as st
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, ArrayType
from pyspark.sql.functions import col, when, collect_list
from pyspark.ml.feature import CountVectorizer, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

# 1. Initialize Spark & Train Model (Cached so it only runs once at startup)
@st.cache_resource
def initialize_system():
    spark = SparkSession.builder \
        .appName("Automated_ER_Triage_UI") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # Load data (Assuming files are in the same directory)
    patients_df = spark.read.csv("patients.csv", header=True, inferSchema=True)
    patient_evidences_df = spark.read.csv("patient_evidences.csv", header=True, inferSchema=True)
    diseases_df = spark.read.csv("diseases.csv", header=True, inferSchema=True)
    
    # Prepare target variable
    model_df = patients_df.join(diseases_df, patients_df.ground_truth_disease_id == diseases_df.disease_id, "left")
    model_df = model_df.withColumn(
        "Triage_Priority",
        when(col("severity") >= 4, "RED").when(col("severity") == 3, "YELLOW").otherwise("GREEN")
    )
    
    # Aggregate symptoms
    symptoms_agg = patient_evidences_df.groupBy("patient_id").agg(collect_list("evidence_id").alias("symptoms_array"))
    master_df = model_df.join(symptoms_agg, on="patient_id", how="inner")
    
    # Build ML Pipeline
    vectorizer = CountVectorizer(inputCol="symptoms_array", outputCol="features")
    indexer = StringIndexer(inputCol="Triage_Priority", outputCol="label")
    rf_classifier = RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=50)
    
    triage_pipeline = Pipeline(stages=[vectorizer, indexer, rf_classifier])
    
    # Train the model
    trained_model = triage_pipeline.fit(master_df)
    
    # Get a list of all possible symptoms for the UI dropdown
    all_symptoms = [row['evidence_id'] for row in patient_evidences_df.select('evidence_id').distinct().collect()]
    
    return spark, trained_model, all_symptoms

st.set_page_config(page_title="ER Triage AI", layout="centered")

st.title("🚑 Automated ER Triage Optimizer")
st.write("Select patient symptoms below to instantly predict their triage priority.")

# Load the backend system (this will show a spinner while training the first time)
with st.spinner("Initializing PySpark Engine and training model... (This takes a moment on first load)"):
    spark, model, available_symptoms = initialize_system()

# 2. Build the User Interface
st.markdown("---")
patient_name = st.text_input("Patient Name / ID (Optional)")
selected_symptoms = st.multiselect("Select Presenting Symptoms", options=available_symptoms)

if st.button("Generate Triage Priority", type="primary"):
    if not selected_symptoms:
        st.warning("Please select at least one symptom.")
    else:
        # Create a Spark DataFrame from the user's input
        schema = StructType([
            StructField("patient_id", StringType(), True),
            StructField("symptoms_array", ArrayType(StringType()), True)
        ])
        
        input_data = [("new_patient", selected_symptoms)]
        input_df = spark.createDataFrame(input_data, schema=schema)
        
        # Run prediction
        prediction = model.transform(input_df)
        
        # Extract the predicted label index
        predicted_index = prediction.select("prediction").collect()[0][0]
        
        # Map index back to String (RED, YELLOW, GREEN)
        labels = model.stages[1].labels
        triage_result = labels[int(predicted_index)]
        
        # Display the result with appropriate colors
        st.markdown("### Triage Assessment:")
        if triage_result == "RED":
            st.error(f"**PRIORITY: {triage_result} (Immediate Intervention Required)**")
        elif triage_result == "YELLOW":
            st.warning(f"**PRIORITY: {triage_result} (Urgent, but stable)**")
        else:
            st.success(f"**PRIORITY: {triage_result} (Non-Urgent, standard queue)**")