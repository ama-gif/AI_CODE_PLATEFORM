import os
import tempfile
import shutil
from typing import List, Dict
from github import Github
from langchain_community.document_loaders import GitLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.runnables import create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from django.conf import settings
from operator import itemgetter

class RepositoryService:
    @staticmethod
    def load_and_index_repository(repo_url: str, branch: str, file_extensions: List[str], repo_id: int) -> Dict:
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp()
            
            def file_filter(file_path: str) -> bool:
                return any(file_path.endswith(ext) for ext in file_extensions)
            
            loader = GitLoader(
                clone_url=repo_url,
                repo_path=temp_dir,
                branch=branch,
                file_filter=file_filter
            )
            documents = loader.load()
            
            if not documents:
                return {"success": False, "error": "No files found matching the selected extensions"}
            
            language_map = {
                ".py": Language.PYTHON, ".js": Language.JS, ".jsx": Language.JS,
                ".ts": Language.TS, ".tsx": Language.TS, ".java": Language.JAVA,
                ".cpp": Language.CPP, ".c": Language.CPP, ".go": Language.GO, ".rb": Language.RUBY
            }
            
            language = Language.PYTHON
            for ext in file_extensions:
                if ext in language_map:
                    language = language_map[ext]
                    break
            
            text_splitter = RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            vector_store_path = os.path.join(settings.VECTOR_STORE_PATH, f"repo_{repo_id}")
            os.makedirs(vector_store_path, exist_ok=True)
            
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=vector_store_path
            )
            
            return {
                "success": True,
                "file_count": len(documents),
                "chunk_count": len(splits),
                "vector_store_path": vector_store_path
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def get_vector_store(vector_store_path: str):
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return Chroma(persist_directory=vector_store_path, embedding_function=embeddings)


class ChatService:
    @staticmethod
    def get_response(question: str, vector_store_path: str, chat_history: List[Dict]) -> Dict:
        try:
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0,
                convert_system_message_to_human=True,
                google_api_key=settings.GOOGLE_API_KEY
            )
            
            system_prompt = """You are an AI assistant that helps developers understand code.
Use the following pieces of context from the codebase to answer the question.
If you don't know the answer, just say that you don't know, don't make up an answer.
Provide code examples when relevant and be specific about file locations.

Context from codebase:
{context}"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}")
            ])
            
            vectorstore = RepositoryService.get_vector_store(vector_store_path)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            # Build the chain using LangChain expression language
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            formatted_history = []
            for msg in chat_history:
                if msg['role'] == 'user':
                    formatted_history.append(HumanMessage(content=msg['content']))
                else:
                    formatted_history.append(AIMessage(content=msg['content']))
            
            # Create retrieval chain
            retrieval_chain = (
                {"context": retriever | format_docs, "input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
                | prompt
                | llm
                | StrOutputParser()
            )
            
            response = retrieval_chain.invoke({
                "input": question,
                "chat_history": formatted_history
            })
            
            # Get context documents
            context_docs = []
            retrieved_docs = retriever.get_relevant_documents(question)
            for doc in retrieved_docs[:3]:
                context_docs.append({
                    "content": doc.page_content[:500],
                    "source": doc.metadata.get('source', 'Unknown')
                })
            
            return {
                "success": True,
                "answer": response,
                "context": context_docs
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class GitHubService:
    @staticmethod
    def analyze_issue(repo_name: str, issue_number: int) -> Dict:
        try:
            g = Github(settings.GITHUB_TOKEN)
            repo = g.get_repo(repo_name)
            issue = repo.get_issue(issue_number)
            
            issue_data = {
                "title": issue.title,
                "number": issue.number,
                "state": issue.state,
                "body": issue.body or "No description provided",
                "labels": [label.name for label in issue.labels],
                "comments_count": issue.comments,
                "url": issue.html_url,
                "user": issue.user.login,
                "created_at": issue.created_at,
                "updated_at": issue.updated_at
            }
            
            comments = []
            for comment in issue.get_comments()[:5]:
                comments.append({
                    "user": comment.user.login,
                    "body": comment.body,
                    "created_at": str(comment.created_at)
                })
            issue_data["comments"] = comments
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.3,
                convert_system_message_to_human=True,
                google_api_key=settings.GOOGLE_API_KEY
            )
            
            analysis_prompt = f"""Analyze this GitHub issue and provide:
1. A brief summary of the problem
2. Possible root causes
3. Suggested solutions or next steps
4. Priority level (High/Medium/Low)

Issue Title: {issue_data['title']}
Issue Body: {issue_data['body'][:1000]}
Labels: {', '.join(issue_data['labels']) if issue_data['labels'] else 'None'}
State: {issue_data['state']}"""
            
            analysis = llm.invoke([HumanMessage(content=analysis_prompt)]).content
            issue_data["ai_analysis"] = analysis
            
            return {"success": True, "data": issue_data}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def search_issues(repo_name: str, keywords: str, labels: str, state: str, max_results: int) -> Dict:
        try:
            g = Github(settings.GITHUB_TOKEN)
            repo = g.get_repo(repo_name)
            
            issues_list = []
            label_list = [l.strip() for l in labels.split(",")] if labels else []
            
            for issue in repo.get_issues(state=state, labels=label_list):
                if len(issues_list) >= max_results:
                    break
                
                if keywords:
                    keyword_list = keywords.lower().split()
                    issue_text = f"{issue.title} {issue.body or ''}".lower()
                    if not any(kw in issue_text for kw in keyword_list):
                        continue
                
                issues_list.append({
                    "number": issue.number,
                    "title": issue.title,
                    "state": issue.state,
                    "labels": [label.name for label in issue.labels],
                    "created_at": str(issue.created_at),
                    "comments": issue.comments,
                    "url": issue.html_url
                })
            
            return {"success": True, "issues": issues_list}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
        