import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Cargar variables de entorno
load_dotenv()

# Definir la carpeta de destino de manera global
CARPETA_DESTINO = "ximenez_vectors"


def get_conversational_chain():
    """Crea una cadena de preguntas y respuestas usando GIANA, la Asistente de Gobierno de Daniel Ximénez."""
    prompt_template = """
    Eres GIANA, la Asistente de Gobierno de Daniel Ximénez. Tu misión es ayudar a los usuarios a conocer las propuestas de gobierno de Daniel Ximénez.
    Responde a las preguntas de los usuarios de manera detallada y amigable utilizando el contexto proporcionado.

    Reglas:
    - Si el usuario pregunta "¿quién es Daniel Ximénez?", responde "Médico Cirujano, Padre de Familia y el próximo intendente de Lavalleja".
    - Si el usuario pregunta "¿quién es tú?", responde "Mi nombre es GIANA y soy la Asistente de Gobierno de Daniel Ximénez, estoy aquí para responder a todas tus preguntas sobre las Propuestas de Gobierno de Daniel".
    - Solo responde preguntas relacionadas con las propuestas de gobierno de Daniel Ximénez.
    - Si la pregunta está fuera de este tema, responde: "Lo siento, solo puedo ayudarte con preguntas sobre las propuestas de gobierno de Daniel Ximénez".

    Si la respuesta no se encuentra en el contexto proporcionado, simplemente di: "Estamos trabajando en ello, pronto tendremos la información que solicitaste sobre ese punto de gobierno".
    No proporciones información incorrecta.

    Contexto:
    {context}

    Pregunta:
    {question}

    Respuesta:
    """

    # Configuración del modelo generativo
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question, vector_store):
    """Procesa la pregunta del usuario y busca respuestas en la base de datos vectorial."""
    # Verificar preguntas específicas
    pregunta_lower = user_question.lower().strip()
    if pregunta_lower == "¿quién es daniel ximénez?" or pregunta_lower == "quién es daniel ximénez":
        st.write("**Respuesta:** Médico Cirujano, Padre de Familia y el próximo intendente de Lavalleja.")
        return
    elif pregunta_lower == "¿quién es tú?" or pregunta_lower == "quién es tú":
        st.write("**Respuesta:** Mi nombre es GIANA y soy la Asistente de Gobierno de Daniel Ximénez, estoy aquí para responder a todas tus preguntas sobre las Propuestas de Gobierno de Daniel.")
        return

    # Si no es una pregunta específica, proceder con la búsqueda
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("**Respuesta:**", response["output_text"])


def main():
    """Interfaz principal de la aplicación con Streamlit como Asistente de Gobierno de Daniel Ximénez."""
    st.set_page_config(page_title="GIANA - Asistente de Gobierno | Daniel Ximénez", layout="wide")
    st.title("🤖 GIANA - Asistente de Gobierno de Daniel Ximénez")

    st.write("""
    ¡Bienvenido! Soy **GIANA**, tu asistente de gobierno. Estoy aquí para responder todas tus preguntas sobre las Propuestas de Gobierno de Daniel Ximénez para transformar Lavalleja.
    """)

    user_question = st.text_input("¿En qué puedo ayudarte hoy?")

    # Cargar la base de datos vectorial al iniciar la aplicación
    if 'vector_store' not in st.session_state:
        with st.spinner("Cargando las propuestas de gobierno de Daniel Ximénez..."):
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            try:
                # Ruta completa para cargar el índice FAISS desde "ximenez_vectors"
                ruta_faiss = os.path.join(CARPETA_DESTINO, "faiss_index")
                vector_store = FAISS.load_local(ruta_faiss, embeddings, allow_dangerous_deserialization=True)
                st.session_state.vector_store = vector_store
                st.success("Propuestas de gobierno cargadas correctamente.")
            except Exception as e:
                st.error(f"Error al cargar la base de datos vectorial: {e}")

    if user_question and 'vector_store' in st.session_state:
        user_input(user_question, st.session_state.vector_store)

    with st.sidebar:
        st.header("✨ Acerca de GIANA")
        st.write("""
        GIANA es tu asistente de gobierno personalizada, diseñada para proporcionarte información detallada y precisa sobre las propuestas de gobierno de Daniel Ximénez.
        Descubre cómo planeamos transformar Lavalleja juntos.
        """)

        st.header("📚 Información")
        st.write("""
        Aprende más sobre las iniciativas de Daniel Ximénez, desde salud y educación hasta desarrollo sostenible y empleo.
        Explora nuestras propuestas para construir un Lavalleja próspero y moderno.
        """)

        st.header("🔗 Conoce Nuestra Plataforma de Gobierno")
        st.write("""
        - [Plataforma de Gobierno](https://cdn.jsdelivr.net/gh/NebyX1/test-program-lol@main/Plataforma%20Program%C3%A1tica%20Lavalleja%20S%C3%AD%21.pdf)
        """)


if __name__ == "__main__":
    main()
