# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("rag-chat/", views.chat_page, name="rag_chat_page"),
    path("api/rag-chat/", views.rag_chat_api, name="rag_chat_api"),
    path("api/tts/", views.tts_api, name="tts_api"),
    path("api/rss-upload/", views.upload_rss_feed, name="rss_upload_api"),
    path("api/url-upload/", views.upload_single_url, name="url_upload_api"),
    path("api/vector-store/rebuild/", views.vector_store_rebuild_api, name="vector_store_rebuild_api"),
    path("api/vector-stores/list/", views.list_vector_stores_api, name="list_vector_stores_api"),
    path("api/upload/files/", views.upload_files, name="upload_files_to_vectorstore_api"),
]
