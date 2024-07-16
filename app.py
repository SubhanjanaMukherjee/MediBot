import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from difflib import SequenceMatcher

def main():
    st.markdown("<h1 style='text-align: left; color: white;'>MediBot ⚕️</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: left; font-style: italic; font-size: 1.2em; color: white;'>Your Medical Assistant</h2>", unsafe_allow_html=True)
    
    page_options = ["Chat with me", "Wellness Check", "Health Condition Evaluation", "Symptom Diagnosis Assistant", "Health Packages"]
    page_selection = st.sidebar.radio("Select an option:", page_options)

    # Load PDF files from the path
    loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                       model_kwargs={'device': "cpu"})

    # Vectorstore
    vector_store = FAISS.from_documents(text_chunks, embeddings)

    # Create LLM
    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama",
                        config={'max_new_tokens': 128, 'temperature': 0.01})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory=memory)

    def is_follow_up_question(current_input, past_questions, threshold=0.6):
        """
        Check if the current input is a follow-up question based on similarity to past questions.

        Args:
        current_input (str): The current user input.
        past_questions (list): List of past user questions.
        threshold (float): Similarity threshold for considering an input as a follow-up question.

        Returns:
        bool: True if the current input is a follow-up question, False otherwise.
        """
        for past_question in past_questions:
            similarity_ratio = SequenceMatcher(None, current_input.lower(), past_question.lower()).ratio()
            if similarity_ratio > threshold:
                return True
        return False

    if page_selection == "Chat with me":
        def conversation_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]

        def initialize_session_state():
            if 'history' not in st.session_state:
                st.session_state['history'] = []

            if 'generated' not in st.session_state:
                st.session_state['generated'] = ["Hello! Ask me anything about your health🤗"]

            if 'past' not in st.session_state:
                st.session_state['past'] = ["Hey! 👋"]

        def display_chat_history():
            reply_container = st.container()
            container = st.container()

            with container:
                with st.form(key='my_form', clear_on_submit=True):
                    user_input = st.text_input("Question:", placeholder="Ask about your Health", key='input')
                    submit_button = st.form_submit_button(label='Send')

                if submit_button and user_input:
                    if is_follow_up_question(user_input, st.session_state['past']):
                        context = st.session_state['past'][-1]  # Use the latest question as context
                    else:
                        context = None

                    output = conversation_chat(user_input)

                    st.session_state['past'].append(user_input)
                    st.session_state['generated'].append(output)

            if st.session_state['generated']:
                with reply_container:
                    for i in range(len(st.session_state['generated'])):
                        message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                        message(st.session_state["generated"][i], key=str(i), avatar_style="fun-emoji")

        # Initialize session state
        initialize_session_state()
        # Display chat history
        display_chat_history()

    elif page_selection == "Wellness Check":
            def conversation_chat(query):
                result = chain({"question": query})
                return result["answer"]


            def display_chat_history_tests():
                reply_container = st.container()


                # Display form
                with st.form(key='my_form', clear_on_submit=True):
                    age = st.radio("Select your age range:", ["18-39", "40-64", "65 and above"], key='age_input')
                    gender = st.radio("Select your gender:", ["Male", "Female"], key='gender_input')
                    smoker = st.radio("Are you a smoker?", ["Yes", "No"], key='smoker_input')
                    #lifestage = st.radio("What lifestage are you in?", ["Child", "Adolescence", "Adult", "Old Age"], key='lifestage_input')
                    submit_button = st.form_submit_button(label='Send')


                if submit_button:
                    user_query = f"What are the tests recommended for {gender} in the age range of {age}?"
                    output = conversation_chat(user_query)
                    # Display response
                    st.write(output)

            # Display chat history
            display_chat_history_tests()


    elif page_selection == "Health Packages":
        st.markdown("<h2 style='text-align: left; color: white;'>Health Packages</h2>", unsafe_allow_html=True)

        # Form to input patient details
        st.subheader("Patient Details")
        name = st.text_input("Name:")
        age = st.text_input("Age:")
        gender = st.selectbox("Gender:", ["Male", "Female", "Other"])

        #if st.button("Submit"):
            #st.success("Patient details submitted successfully!")

        st.markdown("---")  # Separator line

        # Health packages
        st.subheader("Select Health Packages")

        health_packages = {
            "Basic Health Check-up Package": {
                "price": 1000,
                "tests": [
                    "Blood pressure measurement",
                    "Body mass index (BMI) calculation",
                    "Complete blood count (CBC)",
                    "Fasting blood sugar (FBS) test",
                    "Lipid profile (cholesterol) test",
                    "Urinalysis"
                ]
            },
            "Comprehensive Wellness Package": {
                "price": 15000,
                "tests": [
                    "All tests from the Basic Health Check-up Package",
                    "Liver function tests (LFTs)",
                    "Kidney function tests (KFTs)",
                    "Thyroid function tests (TFTs)",
                    "Electrocardiogram (ECG)",
                    "Chest X-ray"
                ]
            },
            "Executive Health Screening Package": {
                "price": 22500,
                "tests": [
                    "All tests from the Comprehensive Wellness Package",
                    "Additional cardiovascular assessments such as stress test or echocardiogram",
                    "Additional screening tests based on individual risk factors and medical history"
                ]
            },
            "Cardiac Risk Assessment Package": {
                "price": 20000,
                "tests": [
                    "All tests from the Comprehensive Wellness Package",
                    "Additional focus on cardiac health, including specialized cardiac markers and imaging tests like coronary CT angiography or cardiac MRI"
                ]
            },
            "Women's Health Package": {
                "price": 11250,
                "tests": [
                    "All tests from the Comprehensive Wellness Package",
                    "Gynecological examination",
                    "Pap smear (cervical cancer screening)",
                    "Breast examination",
                    "Mammogram (for women above a certain age or with specific risk factors)"
                ]
            },
            "Men's Health Package": {
                "price": 11250,
                "tests": [
                    "All tests from the Comprehensive Wellness Package",
                    "Prostate-specific antigen (PSA) test (for prostate cancer screening)",
                    "Testicular examination",
                    "Other tests as recommended based on individual health history and risk factors"
                ]
            }
        }

        selected_packages = []

        total_amount = 0

        # Display packages
        for package_name, package_details in health_packages.items():
            if st.checkbox("", key=f"{package_name}_checkbox"):
                selected_packages.append(package_name)
                total_amount += package_details["price"]

            st.markdown(
                f"""
                <div onclick="toggleCheckbox('{package_name}_checkbox')" style='border: 2px solid white; border-radius: 10px; padding: 10px; margin-bottom: 10px; cursor: pointer;'>
                    <h3 style='color: white;'>{package_name}</h3>
                    <p style='color: white;'>Price: ₹{package_details['price']}</p>
                    <p style='color: white;'><strong>Tests:</strong></p>
                    <ul style='color: white;'>
                        {''.join([f'<li>{test}</li>' for test in package_details['tests']])}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        if selected_packages:
            st.markdown(f"<h3 style='color: white;'>Total Amount: ₹{total_amount}</h3>", unsafe_allow_html=True)

        if st.button("Submit"):
            if selected_packages:
                st.success("Package(s) booked! Please visit the health center according to your slot.")
            else:
                st.warning("Please select at least one package before submitting.")

    elif page_selection == "Health Condition Evaluation":

        def conversation_chat(query):
            result = chain({"question": query})
            return result["answer"]
        
        def display_chat_history_tests():
            reply_container = st.container()

            # Display form
            with st.form(key='my_form', clear_on_submit=True):
                condition = st.radio("Select your medical condition:", ["Pre-diabetes", "Hyperthyroidism", "Hypothyroidism", "Heart Diseases", "Osteoporosis", "PCOS"], key='condition_input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button:
                user_query = f"What are the tests recommended for person with {condition} health condition?"
                output = conversation_chat(user_query)
                # Display response
                st.write(output)
        
        # Display chat history
        display_chat_history_tests()

    elif page_selection == "Symptom Diagnosis Assistant":
        def conversation_chat(query):
            result = chain({"question": query})
            return result["answer"]
        
        def display_chat_history_tests():
            reply_container = st.container()

            # Display form
            with st.form(key='my_form', clear_on_submit=True):
                st.write("Select the symptoms you are experiencing:")
                symptoms = st.multiselect("Symptoms:", ["Darkened skin", "Increased thirst", "Frequent urination", "Increased hunger",
                                                        "Fatigue", "Blurred vision", "Numbness or tingling in the feet or hands",
                                                        "Frequent infections", "Slow-healing sores", "Unintended weight loss",
                                                        "Anxiety", "Difficulty concentrating", "Frequent bowel movements",
                                                        "Goiter (enlarged thyroid gland) or thyroid nodules", "Hair loss",
                                                        "Hand tremor", "Heat intolerance", "Increased appetite", "Increased sweating",
                                                        "Irregular menstrual periods (in women)", "Nail changes", "Nervousness",
                                                        "Palpitations", "Restlessness", "Sleep problems", "Weight changes",
                                                        "Tiredness", "Sensitivity to cold", "Constipation", "Dry skin",
                                                        "Puffy face", "Hoarse voice", "Coarse hair and skin", "Muscle weakness",
                                                        "Muscle aches", "Tenderness and stiffness", "Thinning hair",
                                                        "Slowed heart rate (bradycardia)", "Depression", "Memory problems",
                                                        "Chest pain", "Tightness or pressure in the chest", "Shortness of breath",
                                                        "Pain in the neck, jaw, throat, upper belly area, or back", "Numbness, weakness, or coldness in the limbs due to narrowed blood vessels",
                                                        "Dizziness", "Fainting", "Fluttering or racing heartbeat (palpitations)",
                                                        "Lightheadedness", "Swelling", "Back pain", "Loss of height",
                                                        "A stooped posture", "Easily broken bones", "Reduced bone density",
                                                        "Irregular menstrual cycles (in women)", "Hormonal imbalances",
                                                        "Fertility issues"])
                submit_button = st.form_submit_button(label='Send')

            if submit_button:
                user_query = f"What could be my condition based on the symptoms: {', '.join(symptoms)}?"
                output = conversation_chat(user_query)
                # Display response
                st.write(output)
        
        # Display chat history
        display_chat_history_tests()


if __name__ == "__main__":
    main()
