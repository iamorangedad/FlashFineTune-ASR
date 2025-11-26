import torch
import torch.nn as nn
from model.asr import ASRModel

MAX_LEN = 50
START_TOKEN = 1
END_TOKEN = 2


def inference(model, audio):
    with torch.no_grad():
        enc_output = model.encode(audio)
    max_len = MAX_LEN
    start_token = START_TOKEN
    end_token = END_TOKEN
    decoded = [start_token]
    with torch.no_grad():
        for _ in range(max_len):
            tgt = torch.tensor([decoded]).long()
            tgt_mask = (
                model.generate_square_subsequent_mask(len(decoded))
                .unsqueeze(0)
                .unsqueeze(0)
            )

            output = model.decode(tgt, enc_output, tgt_mask=tgt_mask)
            next_token = output[0, -1, :].argmax().item()

            decoded.append(next_token)
            if next_token == end_token:
                break
    return decoded


def test_model_online_learning():
    N_MELS = 20
    VOCAB_SIZE = 100
    model = ASRModel(
        n_mels=N_MELS,
        d_model=64,
        n_heads=2,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=128,
        vocab_size=VOCAB_SIZE,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    accumulation_steps = 10
    total_loss = 0.0
    for step in range(accumulation_steps):
        model.eval()
        # t0 inference
        audio_t0 = torch.randn(1, 10, N_MELS)
        decoded_t0 = inference(model, audio_t0)
        # t1 inference
        audio_t1 = torch.randn(1, 10, N_MELS)
        decoded_t1 = inference(model, audio_t1)
        # t0 & t1 train
        model.train()
        text_len = len(decoded_t1) - 1
        text = torch.tensor(decoded_t1).unsqueeze(0)
        tgt_input = text[:, :-1]
        tgt_output = text[:, 1:]
        tgt_mask = (
            model.generate_square_subsequent_mask(text_len).unsqueeze(0).unsqueeze(0)
        )
        output = model(audio_t0, tgt_input, tgt_mask=tgt_mask)
        loss = criterion(output.reshape(-1, VOCAB_SIZE), tgt_output.reshape(-1))
        (loss / accumulation_steps).backward()
        total_loss += loss.item()
    optimizer.step()
    optimizer.zero_grad()
    assert isinstance(total_loss, float)
