import torch
import torch.nn as nn


def make_conv_block(in_c, out_c, kernel_size=(3, 5, 5), padding=(0, 2, 2)):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm3d(out_c),
        nn.ReLU()
    )

# a classifier that takes a batch of (consecutive) frames as input and outputs 
# whether it thinks those frames are part of an AI-generated video or not
#
# currently takes 10 frames, changing that could require some architectural 
# changes (e.g. using pooling on the temporal dimension instead of just spatial 
# ones, or changing number of layers/kernel sizes)
class AIClassifier(nn.Module):
    def __init__(self, frames=10, frame_dim=(500, 500), init_channels=16, head_drop=0.2):
        super(AIClassifier, self).__init__()
        self.frames = frames
        self.width = frame_dim[0]
        self.height = frame_dim[1]
        self.img_avg_pool = nn.AvgPool3d((1, 2, 2), stride=(1, 2, 2))
        self.img_component = nn.Sequential(
            make_conv_block(3, init_channels), 
            self.img_avg_pool,
            # B x 16 x 8 x 250 x 250
            make_conv_block(init_channels, init_channels * 2), 
            self.img_avg_pool,
            # B x 32 x 6 x 125 x 125
            make_conv_block(init_channels * 2, init_channels * 4),
            self.img_avg_pool,
            # B x 64 x 4 x 62 x 62
            make_conv_block(init_channels * 4, init_channels * 8),
            self.img_avg_pool,
            # B x 128 x 2 x 31 x 31
            make_conv_block(init_channels * 8, init_channels * 16, kernel_size=(2, 5, 5)),
            self.img_avg_pool,
            # B x 256 x 1 x 15 x 15
            nn.Flatten(),
            # B x 57,600
        )
        self.classifier = nn.Sequential(
            nn.Linear(init_channels * 16 * (15**2), 2**12),
            nn.ReLU(),
            nn.Dropout(head_drop),
            nn.Linear(2**12, 2**9),
            nn.ReLU(),
            nn.Dropout(head_drop),
            nn.Linear(2**9, 2**6),
            nn.ReLU(),
            nn.Dropout(head_drop),
            nn.Linear(2**6, 1),
            nn.Sigmoid()
        )
    
    def forward(self, frames):
        # frames: B x C x F x W x H
        features = self.img_component(frames)
        return self.classifier(features)



# check with dummy data 
if __name__ == "__main__":
    print("start")
    model = AIClassifier()
    print("Model created")
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    dummy = torch.randn(2, 3, 10, 500, 500)
    print("Predicting")
    out = model(dummy)
    print("Done predicting")
    labels = torch.Tensor([0.0, 1.0]).unsqueeze(1)
    loss = criterion(out, labels)
    print(f"Output shape: {out.shape}") 
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Img params: {sum(p.numel() for p in model.img_component.parameters()):,}")
    print(f"Head params: {sum(p.numel() for p in model.classifier.parameters()):,}")
    print(f"Loss: {loss.item():.4f}")