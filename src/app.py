
from langchain_groq import ChatGroq
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import shutil
import gradio as gr
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
import re
import gc
import shutil

# --- CHEMINS ---

NEW_DATA_PATH = "./nouveau_data"
ACTUEL_DATA_PATH = "./actuel_data"
CHROMA_PATH = "./chroma_db"

# Création des dossiers si nécessaire
for path in [NEW_DATA_PATH, ACTUEL_DATA_PATH, CHROMA_PATH]:
    os.makedirs(path, exist_ok=True)

# --- CONFIGURATION API ---

os.environ["GROQ_API_KEY"] = os.getenv("HR_Assistant_API")
llm_chat = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.5
)

llm_multi_query = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# Le rédacteur
instruction_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Tu es un expert en analyse de documents RH. Ton rôle est de fournir des réponses "
        "précises et structurées à partir du contexte fourni ci-dessous.\n\n"
        "CONTEXTE :\n{context}\n\n"
        "DIRECTIVES DE PRÉCISION :\n"
        "1. SYNTHÈSE : Synthétise les points clés (noms, dates, montants).\n"
        "2. SOURCES : Pour chaque information importante, tu DOIS citer le document source. "
        "Exemple : 'La prime est de 300€ [Source: politique_primes.pdf]'.\n"
        "3. INTÉGRITÉ : Si l'info n'est pas dans le contexte, dis 'Je ne trouve pas cette info dans les documents fournis'.\n"
        "4. ANTI-MÉLANGE : Si deux documents se contredisent, précise-le (ex: 'Le doc A dit X, mais le doc B dit Y').\n"
        "5. FORMAT : Utilise des listes à puces et du gras pour les chiffres."
    )),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def get_or_update_db(update=True):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 1. Charger la base existante (ou en créer une vide)
    vector_db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # 2. Vérifier s'il y a du nouveau
    if update:
      new_files = [f for f in os.listdir(NEW_DATA_PATH) if f.endswith('.pdf')]

      if new_files:
          loader = PyPDFDirectoryLoader(NEW_DATA_PATH)
          new_docs = loader.load()

          text_splitter = RecursiveCharacterTextSplitter(
              chunk_size=800,
              chunk_overlap=200,
              length_function=len,
              add_start_index=True
          )

          new_chunks = text_splitter.split_documents(new_docs)

          # Ajout à Chroma
          vector_db.add_documents(new_chunks)

          # Déplacement vers l'archive
          for file_name in new_files:
              shutil.move(os.path.join(NEW_DATA_PATH, file_name),
                          os.path.join(ACTUEL_DATA_PATH, file_name))
              print(f"Archivé : {file_name}")
      else:
          print("Aucun nouveau fichier. Utilisation de la base existante.")

    return vector_db

def vider_db():
    # Vider la collection
    try:
        vector_db = get_or_update_db(False)
        vector_db.delete_collection()
        print("Collection vidée.")

    except Exception as e:
        print(f"Erreur lors du vidage : {e}")

    # Supprimer la collection
    for nom in os.listdir(CHROMA_PATH):
        item_path = os.path.join(path, nom)
        if os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path)
                print(f"Dossier de collection supprimé : {nom}")
            except Exception as e:
                print(f"Impossible de supprimer {nom} : {e}")

    # Supprimer les fichiers PDF dans actuel_data
    for nom in os.listdir(ACTUEL_DATA_PATH):
        chemin_complet = os.path.join(ACTUEL_DATA_PATH, nom)
        try:
            if os.path.isfile(chemin_complet):
                os.unlink(chemin_complet)
            elif os.path.isdir(chemin_complet):
                shutil.rmtree(chemin_complet)
            print("Actuel data vidé")
        except Exception as e:
            print(f"Impossible de supprimer {nom} : {e}")
    return "✅ La base de données a été vidée avec succès."

def respond(message, chat_history):
    global llm_chat, llm_multi_query

    # 1. Récupération de la base
    vector_db = get_or_update_db(False)

    # --- TRANSFORME L'HISTORIQUE POUR LE LLM ---
    #formatted_history = []
    #for user_msg, ai_msg in chat_history:
        #formatted_history.append(HumanMessage(content=user_msg))
        #formatted_history.append(AIMessage(content=ai_msg))
        #formatted_history.append({"role": "user", "content": str(user_msg)})
        #formatted_history.append({"role": "assistant", "content": str(ai_msg)})
        
    # --- ÉTAPE 2 : RECHERCHE AVANCÉE (Multi-Query) ---
    advanced_retriever = MultiQueryRetriever.from_llm(
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        llm=llm_multi_query
    )
    
    
    # Le Multi-Query va générer des variantes pour mieux fouiller la DB
    docs = advanced_retriever.invoke(input=message)
    context = "\n\n".join([d.page_content for d in docs])

    # --- ÉTAPE 3 : RÉPONSE FINALE (Le Rédacteur) ---
    chain = instruction_prompt | llm_chat
    result = chain.invoke({
        "context": context,
        "question": message,
        "chat_history": chat_history
    })
    answer = result.content if hasattr(result, 'content') else str(result)

    # 4. Ajouter l'échange à l'historique
    chat_history.append({"role": "user", "content": message})
    chat_history.append({"role": "assistant", "content": answer})

    return "", chat_history

def upload_files(files):
    # 1. Sécurité anti-vide
    if not files:
        gr.Warning("⚠️ Veuillez sélectionner au moins un fichier.")
        return "ERREUR : Aucun fichier sélectionné."

    # récuperer les noms de fichiers déjà archivés
    actuel_files = set(os.listdir(ACTUEL_DATA_PATH))

    files_to_process = []
    doublons_count = 0

    for file in files:
        # extraire le nom propre du fichier (ex: "charte.pdf")
        nom_fichier = os.path.basename(file.name)

        # COMPARAISON : vérifier si le NOM est déjà dans le SET
        if nom_fichier not in actuel_files:
            destination = os.path.join(NEW_DATA_PATH, nom_fichier)
            shutil.copy(file.name, destination)
            files_to_process.append(nom_fichier)
        else:
            doublons_count += 1

    # 2. ne lancer que s'il y a du nouveau fichier
    if files_to_process:
        get_or_update_db(update=True)
        msg = f"✅ SUCCÈS : {len(files_to_process)} fichiers ajoutés."
        if doublons_count > 0:
            msg += f" ({doublons_count} doublons ignorés)"
        return msg
    else:
        return "ℹ️ Tous ces fichiers sont déjà dans la base. Aucun nouveau fichier à synchroniser."

import base64
PATH_TO_ICON = "./icon/chatbot.png"

def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    else:
        print(f"⚠️ Erreur : Fichier non trouvé à l'adresse {image_path}")
        return ""

img_data = get_base64_image(PATH_TO_ICON)

with gr.Blocks() as demo:
    gr.HTML(f"""
        <div style="display: flex; align-items: center; gap: 15px; padding: 10px;">
            <img src="data:image/png;base64,{img_data}" width="50px">
            <h1 style="margin: 0; font-family: sans-serif;">Assistant RH Intelligent</h1>
        </div>
    """)

    with gr.Tabs():
        # --- ONGLET 1 : CHAT ---
        with gr.Tab("💬 Chat avec l'Expert"):
            chatbot = gr.Chatbot(label="Discussion", height=500)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Posez votre question (ex: Quelles sont les règles de mutuelle ?)",
                    show_label=False,
                    scale=9
                )
                submit_btn = gr.Button("Envoyer", variant="primary", scale=1)

            clear_chat = gr.ClearButton([msg, chatbot], value="Effacer la conversation")

        # --- ONGLET 2 : ADMINISTRATION ---
        with gr.Tab("⚙️ Administration"):
            gr.Markdown("### 📂 Gestion des documents")
            gr.Markdown("Ajoutez de nouveaux PDF pour mettre à jour la base de connaissances.")

            with gr.Row():
                upload_btn = gr.File(
                    label="Déposez vos PDF ici",
                    file_count="multiple",
                    file_types=[".pdf"]
                )

            # Bouton de validation principal
            process_btn = gr.Button("📥 Ajouter à la base de données", variant="primary")

            # Afficheur de statut
            status_label = gr.Textbox(
                              label="Statut du traitement",
                              placeholder="En attente...",
                              lines=7,
                              max_lines=10,
                              interactive=False,
                              show_label=True
                          )
            gr.HTML("<br><br><br>") # Un peu d'espace avant la zone de danger
            gr.Markdown("---")

            # Zone de danger en bas à gauche
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("🔒 **Maintenance**")
                    clear_db_btn = gr.Button("🗑️ Vider la base de données", variant="stop", size="sm")
                with gr.Column(scale=4):
                    # Cette colonne vide sert à pousser le bouton vers la gauche
                    pass

    # --- LOGIQUE DES BOUTONS ---

    # Envoi du message
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit_btn.click(respond, [msg, chatbot], [msg, chatbot])

    # Indexation des fichiers
    process_btn.click(
        fn=upload_files,
        inputs=upload_btn,
        outputs=status_label,
        show_progress="full"
    )

    # Vidage de la base
    clear_db_btn.click(
        fn=vider_db,
        outputs=status_label,
        show_progress="full"
    )

# Lancement de l'application
if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), debug=True)