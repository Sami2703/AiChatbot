from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch  # Uncomment if you plan to use GPU

# Load model and tokenizer once when the server starts
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Function to generate chatbot responses
def generate_response(user_input):
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    attention_mask = torch.ones_like(inputs)  # Create attention mask manually
    response_ids = model.generate(inputs,attention_mask=attention_mask, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(response_ids[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return response

# API view for chatbot endpoint
class ChatBotView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        user_input = request.data.get("message")
        if not user_input:
            return Response({"error": "No message provided."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            reply = generate_response(user_input)
            return Response({"response": reply}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
