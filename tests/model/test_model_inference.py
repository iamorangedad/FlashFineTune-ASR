import torch
from model.asr import ASRModel


def test_model_inference():
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
    model.eval()
    audio_features = torch.randn(1, 10, N_MELS)

    with torch.no_grad():
        enc_output = model.encode(audio_features)

    max_len = 50
    start_token = 1
    end_token = 2

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

    assert len(decoded) == (max_len + 1)
