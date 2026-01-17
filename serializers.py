from rest_framework import serializers
from .models import Repository, ChatConversation, ChatMessage, GitHubIssueAnalysis

class RepositorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Repository
        fields = ['id', 'repo_url', 'branch', 'repo_name', 'file_extensions', 
                  'indexed_at', 'updated_at', 'is_active', 'file_count', 'chunk_count']
        read_only_fields = ['indexed_at', 'updated_at', 'file_count', 'chunk_count']


class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['id', 'role', 'content', 'context_documents', 'created_at']
        read_only_fields = ['created_at']


class ChatConversationSerializer(serializers.ModelSerializer):
    messages = ChatMessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = ChatConversation
        fields = ['id', 'repository', 'title', 'created_at', 'updated_at', 'messages', 'message_count']
        read_only_fields = ['created_at', 'updated_at']

    def get_message_count(self, obj):
        return obj.messages.count()


class GitHubIssueAnalysisSerializer(serializers.ModelSerializer):
    class Meta:
        model = GitHubIssueAnalysis
        fields = ['id', 'repo_name', 'issue_number', 'title', 'state', 'body', 
                  'labels', 'ai_analysis', 'comments', 'analyzed_at', 'issue_url']
        read_only_fields = ['analyzed_at']


class ChatQuerySerializer(serializers.Serializer):
    conversation_id = serializers.IntegerField(required=True)
    question = serializers.CharField(required=True, max_length=2000)


class RepositoryIndexSerializer(serializers.Serializer):
    repo_url = serializers.URLField(required=True)
    branch = serializers.CharField(default='main', max_length=100)
    file_extensions = serializers.ListField(
        child=serializers.CharField(max_length=10),
        default=['.py']
    )


class IssueAnalysisRequestSerializer(serializers.Serializer):
    repo_name = serializers.CharField(required=True, max_length=255)
    issue_number = serializers.IntegerField(required=True, min_value=1)


class IssueSearchSerializer(serializers.Serializer):
    repo_name = serializers.CharField(required=True, max_length=255)
    keywords = serializers.CharField(required=False, allow_blank=True)
    labels = serializers.CharField(required=False, allow_blank=True)
    state = serializers.ChoiceField(choices=['open', 'closed', 'all'], default='open')
    max_results = serializers.IntegerField(default=10, min_value=1, max_value=100)