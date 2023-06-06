import re
import textgrid
import torch
from torch import nn


from transformers import (Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor)

def parse_Interval(IntervalObject):
    start_time = ""
    end_time = ""
    P_name = ""

    ind = 0
    str_interval = str(IntervalObject)
    for ele in str_interval:
        if ele == "(":
            ind = 1
        if ele == " " and ind == 1:
            ind = 2
        if ele == "," and ind == 2:
            ind = 3
        if ele == " " and ind == 3:
            ind = 4

        if ind == 1:
            if ele != "(" and ele != ",":
                start_time = start_time + ele
        if ind == 2:
            end_time = end_time + ele
        if ind == 4:
            if ele != " " and ele != ")":
                P_name = P_name + ele

    st = float(start_time)
    et = float(end_time)
    pn = P_name

    return (pn, st, et)


def parse_textgrid(filename):
    tg = textgrid.TextGrid.fromFile(filename)
    list_words = tg.getList("words")
    words_list = list_words[0]

    result = []
    for ele in words_list:
        d = parse_Interval(ele)
        result.append(d)
    return result

def create_processor(model_name_or_path,vocab_file= None):
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        model_name_or_path)

    if vocab_file:
        tokenizer = Wav2Vec2CTCTokenizer(
                vocab_file,
                do_lower_case=False,
                word_delimiter_token="|",
            )
    else:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                model_name_or_path,
                do_lower_case=False,
                word_delimiter_token="|",
            )
    return Wav2Vec2Processor(feature_extractor, tokenizer)


def prepare_example(text,vocabulary_text_cleaner):

    # Normalize and clean up text; order matters!
    try:
        text = " ".join(text.split())  # clean up whitespaces
    except:
        text = "NULL"
    updated_text = text
    updated_text = vocabulary_text_cleaner.sub("", updated_text)
    if updated_text != text:
        return re.sub(' +', ' ', updated_text).strip()
    else:
        return re.sub(' +', ' ', text).strip()

class StatisticsPooling(nn.Module):
    """This class implements a statistic pooling layer.
    It returns the mean and/or std of input tensor.
    Arguments
    ---------
    return_mean : True
         If True, the average pooling will be returned.
    return_std : True
         If True, the standard deviation will be returned.
    Example
    -------
    >>> inp_tensor = torch.rand([5, 100, 50])
    >>> sp_layer = StatisticsPooling()
    >>> out_tensor = sp_layer(inp_tensor)
    >>> out_tensor.shape
    torch.Size([5, 1, 100])
    """

    def __init__(self, return_mean=True, return_std=True):
        super().__init__()

        # Small value for GaussNoise
        self.eps = 1e-5
        self.return_mean = return_mean
        self.return_std = return_std
        if not (self.return_mean or self.return_std):
            raise ValueError(
                "both of statistics are equal to False \n"
                "consider enabling mean and/or std statistic pooling"
            )

    def forward(self, x, lengths=None):
        """Calculates mean and std for a batch (input tensor).
        Arguments
        ---------
        x : torch.Tensor
            It represents a tensor for a mini-batch.
        """
        if lengths is None:
            if self.return_mean:
                mean = x.mean(dim=1)
            if self.return_std:
                std = x.std(dim=1)
        else:
            mean = []
            std = []
            for snt_id in range(x.shape[0]):
                # Avoiding padded time steps
                actual_size = int(torch.round(lengths[snt_id] * x.shape[1]))

                # computing statistics
                if self.return_mean:
                    mean.append(
                        torch.mean(x[snt_id, 0:actual_size, ...], dim=0)
                    )
                if self.return_std:
                    std.append(torch.std(x[snt_id, 0:actual_size, ...], dim=0))
            if self.return_mean:
                mean = torch.stack(mean)
            if self.return_std:
                std = torch.stack(std)

        if self.return_mean:
            gnoise = self._get_gauss_noise(mean.size(), device=mean.device)
            gnoise = gnoise
            mean += gnoise
        if self.return_std:
            std = std + self.eps

        # Append mean and std of the batch
        if self.return_mean and self.return_std:
            pooled_stats = torch.cat((mean, std), dim=1)
            pooled_stats = pooled_stats.unsqueeze(1)
        elif self.return_mean:
            pooled_stats = mean.unsqueeze(1)
        elif self.return_std:
            pooled_stats = std.unsqueeze(1)

        return pooled_stats

class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 4.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 4.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:2]

def text_preprocessing(text):
    """
    - Remove entity mentions (eg. '@united')
    - Correct errors (eg. '&amp;' to '&')
    @param    text (str): a string to be processed.
    @return   text (Str): the processed string.
    """
    # Remove '@name'
    text = re.sub("[\(\[].*?[\)\]]", '', text)

    # Replace '&amp;' with '&'
    text = re.sub(" +",' ', text).strip()

    return text

def load_checkpoint(model,ckpt_path):

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    for key in checkpoint['state_dict'].copy():
        if not 'transformer' in key:
            del checkpoint['state_dict'][key]
        else:
            new_key = key.lstrip("transformer").lstrip(".")
            checkpoint['state_dict'][new_key] = checkpoint['state_dict'][key]
            del checkpoint['state_dict'][key]

    x,y = model.load_state_dict(checkpoint['state_dict'],strict=False)

    print(x)
    print(y)

    return model

class Classic_Attention(nn.Module):
    def __init__(self,input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim,embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))

    def forward(self,inputs):
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = F.tanh(lin_out.bmm(v_view).squeeze())
        attention_weights_normalized = F.softmax(attention_weights,1)
        return attention_weights_normalized


def downsample(x, x_len, sample_rate=2):

    batch_size, timestep, feature_dim = x.shape
    x_len = x_len // sample_rate
    # Drop the redundant frames and concat the rest according to sample rate
    if timestep % sample_rate != 0:
        x = torch.nn.functional.pad(x,(0,0,0,1))
    if timestep % sample_rate != 0:
        x = x.contiguous().view(batch_size, int(timestep // sample_rate) + 1, feature_dim * sample_rate)
    else:
        x = x.contiguous().view(batch_size, int(timestep // sample_rate), feature_dim * sample_rate)

    return x

def create_mask(batch_size,seq_len,spec_len):

    with torch.no_grad():
        attn_mask = torch.ones((batch_size, seq_len)) # (batch_size, seq_len)

        for idx in range(batch_size):
            # zero vectors for padding dimension
            attn_mask[idx, spec_len[idx]:] = 0

    return attn_mask

class Conv2dSubsampling2(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/2 length).

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.
        pos_enc (torch.nn.Module): Custom position encoding layer.

    """

    def __init__(self, idim, odim, dropout_rate, pos_enc=None):
        """Construct an Conv2dSubsampling2 object."""
        super(Conv2dSubsampling2, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 1),
            torch.nn.ReLU(),
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 2)), odim),
        )

    def forward(self, x, x_mask):
        """Subsample x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: Subsampled tensor (#batch, time', odim),
                where time' = time // 2.
            torch.Tensor: Subsampled mask (#batch, 1, time'),
                where time' = time // 2.

        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask[:, :, :-2:2][:, :, :-2:1]