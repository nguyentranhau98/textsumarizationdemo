import numpy as np
from train_abstractive import test_text_abs
from gui_utils import clean_output

class AbsArgs:
    def __init__(self):
        self.task = 'abs'
        self.mode = 'test_text'
        self.text_src = '/home/nguyentranhau/Desktop/Final/textsumbert/input.txt'
        self.test_from = '/home/nguyentranhau/Desktop/Final/textsumbert/models/abs_model_step_148000.pt'
        self.log_file = '/home/nguyentranhau/Desktop/Final/textsumbert/logs/cnndm.log'
        self.result_path = '/home/nguyentranhau/Desktop/Final/textsumbert/logs'
        self.max_pos = 512
        self.visible_gpus = 0
        self.text_tgt = ''
        self.encoder = 'bert'
        self.bert_data_path = '/home/nguyentranhau/Desktop/Final/textsumbert/bert_data_new/cnndm'
        self.model_path = '/home/nguyentranhau/Desktop/Final/textsumbert/models/'
        self.temp_dir = '/home/nguyentranhau/Desktop/Final/textsumbert/temp'
        self.batch_size = 140
        self.test_batch_size = 200
        self.max_ndocs_in_batch = 6
        self.use_interval = True
        self.large = False
        self.load_from_extractive = '/home/nguyentranhau/Desktop/Final/textsumbert/models/ext_model_step_50000.pt'
        self.sep_optim = True
        self.lr_bert = 0.002
        self.lr_dec = 0.2
        self.use_bert_emb = True
        self.share_emb = False
        self.finetune_bert = True
        self.dec_dropout = 0.2
        self.dec_layers = 6
        self.dec_hidden_size = 768
        self.dec_heads = 8
        self.dec_ff_size = 2048
        self.enc_hidden_size = 512
        self.enc_ff_size = 512
        self.enc_dropout = 0.2
        self.enc_layers = 6

        # params for EXT
        self.ext_dropout = 0.2
        self.ext_layers = 2
        self.ext_hidden_size = 768
        self.ext_heads = 8
        self.ext_ff_size = 2048

        self.label_smoothing = 0.1
        self.generator_shard_size = 32
        self.alpha = 0.6
        self.beam_size = 5
        self.min_length = 15
        self.max_length = 150
        self.max_tgt_len = 140
        self.param_init = 0
        self.param_init_glorot = True
        self.optim = 'adam'
        self.lr = 1
        self.beta1 =  0.9
        self.beta2 = 0.999
        self.warmup_steps = 8000
        self.warmup_steps_bert = 20000
        self.warmup_steps_dec = 10000
        self.max_grad_norm = 0
        self.save_checkpoint_steps = 5
        self.accum_count = 5 
        self.report_every = 1 
        self.train_steps = 1000 
        self.recall_eval = False
        self.gpu_ranks = '0'
        self.world_size = len(self.gpu_ranks)
        self.seed = 666
        self.test_all = False
        self.test_start_from = -1
        self.train_from = ''
        self.report_rouge = True
        self.block_trigram = True

class AbsBert:
    def __init__(self):
        self.args = AbsArgs()
    
    def abs_summarize(self):
        res = ''
        # args = ExtArgs()
        # step = int(self.text_from.split('.')[-2].split('_')[-1])
        # try:
        #     step = int(self.text_from.split('.')[-2].split('_')[-1])
        # except:
        #     step = 0
        test_text_abs(self.args)
        with open('/home/nguyentranhau/Desktop/Final/textsumbert/logs.-1.candidate', 'r') as f:
            res = clean_output(f.readline())
        return res
