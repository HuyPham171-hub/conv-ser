"""
Conversational Bi-GRU Architecture Module.
Defines the sequence modeling network for Contextual Emotion Tracking (Goal 3).
"""

import torch
import torch.nn as nn

class ConversationalBiGRU(nn.Module):
    """
    A Bidirectional GRU network designed to process sequences of acoustic embeddings
    and predict the contextual emotion of the current utterance using a Many-to-One architecture.
    """
    def __init__(self, input_dim=768, hidden_dim=256, num_layers=2, num_classes=3, dropout_rate=0.3):
        """
        Initializes the Bi-GRU model parameters.
        
        Args:
            input_dim (int): Dimension of the input acoustic embeddings (Default: 768 for Wav2Vec2).
            hidden_dim (int): Number of features in the GRU hidden state (Default: 256).
            num_layers (int): Number of recurrent layers (Default: 2 for deeper temporal abstraction).
            num_classes (int): Number of output classes (3 for Stage 1 Sentiment, 5 for Stage 2 Fine-grained).
            dropout_rate (float): Dropout probability to prevent overfitting on small datasets.
        """
        super(ConversationalBiGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # 1. The Core Temporal Engine: Bidirectional GRU
        # batch_first=True ensures input tensor shape is (batch_size, seq_len, input_dim)
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        # 2. Regularization & Normalization
        # LayerNorm stabilizes the learning process across the hidden dimension
        # Multiply hidden_dim by 2 because the GRU is bidirectional (forward + backward states)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 3. The Contextual Arbitrator (Classifier Head)
        # Maps the concatenated bidirectional hidden states to the final emotion classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Defines the forward pass of the network.
        
        Args:
            x (torch.Tensor): Input sequence tensor of shape (batch_size, seq_len=3, input_dim=768).
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        # Pass the sequence through the Bi-GRU
        # gru_out shape: (batch_size, seq_len, hidden_dim * 2)
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)
        gru_out, hidden = self.gru(x)
        
        # MANY-TO-ONE ARCHITECTURE EXTRACTION
        # We only care about the network's understanding at the final time step (the current utterance U_t).
        # We slice the tensor to extract the output at the last sequence index (-1).
        # final_step_out shape: (batch_size, hidden_dim * 2)
        final_step_out = gru_out[:, -1, :]
        
        # Apply normalization and dropout for robust representation
        normalized_out = self.layer_norm(final_step_out)
        regularized_out = self.dropout(normalized_out)
        
        # Compute final class probabilities (logits)
        # logits shape: (batch_size, num_classes)
        logits = self.classifier(regularized_out)
        
        return logits

# =====================================================================
# DRY RUN / SANITY CHECK (Can be executed directly to verify tensor shapes)
# ===================================================================== 
if __name__ == "__main__":
    print("[INFO] Performing Model Architecture Sanity Check...")
    
    # Simulate a dummy batch from the DataLoader
    # Batch Size = 32, Sequence Length N = 3, Acoustic Embedding = 768
    dummy_input = torch.randn(32, 3, 768)
    
    # Instantiate model for Stage 1 (3 classes: Neg, Neu, Pos)
    model_stage1 = ConversationalBiGRU(input_dim=768, hidden_dim=256, num_classes=3)
    
    # Forward pass
    output_stage1 = model_stage1(dummy_input)
    
    print(f"Input Shape       : {dummy_input.shape}")
    print(f"Output Shape (S1) : {output_stage1.shape} -> Expected: [32, 3]")
    print("[SUCCESS] Bi-GRU Architecture is ready for training!")