from django.db import models
from django.contrib.auth.models import User

class Repository(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='repositories')
    repo_url = models.URLField(max_length=500)
    branch = models.CharField(max_length=100, default='main')
    repo_name = models.CharField(max_length=255)
    file_extensions = models.JSONField(default=list)
    vector_store_path = models.CharField(max_length=500, blank=True)
    indexed_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)
    file_count = models.IntegerField(default=0)
    chunk_count = models.IntegerField(default=0)

    class Meta:
        ordering = ['-created_at']
        unique_together = ['user', 'repo_url', 'branch']

    def __str__(self):
        return f"{self.repo_name} ({self.branch})"


class ChatConversation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    repository = models.ForeignKey(Repository, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=255, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"Conversation {self.id} - {self.title or 'Untitled'}"


class ChatMessage(models.Model):
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]

    conversation = models.ForeignKey(ChatConversation, on_delete=models.CASCADE, related_name='messages')
    role = models.CharField(max_length=20, choices=ROLE_CHOICES)
    content = models.TextField()
    context_documents = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"


class GitHubIssueAnalysis(models.Model):
    repository = models.ForeignKey(Repository, on_delete=models.CASCADE, related_name='issue_analyses', null=True, blank=True)
    repo_name = models.CharField(max_length=255)
    issue_number = models.IntegerField()
    title = models.CharField(max_length=500)
    state = models.CharField(max_length=20)
    body = models.TextField(blank=True)
    labels = models.JSONField(default=list)
    ai_analysis = models.TextField(blank=True)
    comments = models.JSONField(default=list)
    analyzed_at = models.DateTimeField(auto_now_add=True)
    issue_url = models.URLField(max_length=500)

    class Meta:
        ordering = ['-analyzed_at']
        unique_together = ['repo_name', 'issue_number']

    def __str__(self):
        return f"#{self.issue_number}: {self.title}"


class APIUsage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='api_usage')
    endpoint = models.CharField(max_length=100)
    method = models.CharField(max_length=10)
    status_code = models.IntegerField()
    response_time = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"{self.user.username} - {self.endpoint} - {self.timestamp}"