from torch import nn

class SKF(nn.Module):
    """
    Add additional module for Patch Embedding loss -> Projectioner for matching chennel between student and teacher
    """
    def __init__(
        self,student, in_channels, out_channels, embed
    ):
        super(SKF, self).__init__()
        self.student = student

        self.embed = embed
        if self.embed == 5:
            self.embed1_linearproject = nn.Linear(in_channels[0], out_channels[0])
            self.embed2_linearproject = nn.Linear(in_channels[1], out_channels[1])
            self.embed3_linearproject = nn.Linear(in_channels[2], out_channels[2])
            self.embed4_linearproject = nn.Linear(in_channels[3], out_channels[3])
        elif self.embed == 1:
            self.embed1_linearproject = nn.Linear(in_channels[0], out_channels[0])
        elif self.embed == 2:
            self.embed1_linearproject = nn.Linear(in_channels[1], out_channels[1])
        elif self.embed == 3:
            self.embed1_linearproject = nn.Linear(in_channels[2], out_channels[2])
        elif self.embed == 4:
            self.embed1_linearproject = nn.Linear(in_channels[3], out_channels[3])

    def forward(self, x):
        student_features = self.student(x,is_feat=True) # "feature maps, logits, patchembeddings -> dictionary"
        features = student_features["feature_map"]
        logit = student_features["logits"]
        embed = student_features["patch_embeddings"]
        embedproj = []
  
        if self.embed ==5: # Return All Stage Patch Embeddings
            embedproj = [*embedproj, self.embed1_linearproject(embed[0])]
            embedproj = [*embedproj, self.embed2_linearproject(embed[1])]
            embedproj = [*embedproj, self.embed3_linearproject(embed[2])]
            embedproj = [*embedproj, self.embed4_linearproject(embed[3])]
            return {"feature_map" : features, "logits" : logit, "patch_embeddings" : embedproj} 
        elif self.embed == 0:
            return {"feature_map" : features, "logits" : logit}
        elif self.embed == 1:
            embedproj = [*embedproj, self.embed1_linearproject(embed[0])]
            return {"feature_map" : features, "logits" : logit, "patch_embeddings" : embedproj}
        elif self.embed == 2:
            embedproj = [*embedproj, self.embed1_linearproject(embed[1])]
            return {"feature_map" : features, "logits" : logit, "patch_embeddings" : embedproj}
        elif self.embed == 3:
            embedproj = [*embedproj, self.embed1_linearproject(embed[2])]
            return {"feature_map" : features, "logits" : logit, "patch_embeddings" : embedproj}
        elif self.embed == 4:
            embedproj = [*embedproj, self.embed1_linearproject(embed[3])]
            return {"feature_map" : features, "logits" : logit, "patch_embeddings" : embedproj}          
        else:
            assert 'the number of embeddings not supported'


def build_kd_trans(model, embed, in_channels = [32, 64, 160, 256], out_channels = [64, 128, 320, 512]):
    """
    Model : Student Model
    Embed : Which stages will attend for Patch embedding loss? => If 5, all stages will attend for Patch Embedding loss
    In Channels : student stage channels
    Out Channels : teacher stage channels
    -> Linearly project for Patch Embedding Loss
    """
    student = model
    model = SKF(student, in_channels, out_channels, embed)
    return model