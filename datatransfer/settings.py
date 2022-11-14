"""
各类参数的默认设定
Tree-like
"""
from datatransfer.Models.RE.RE_settings import duie_relations

"""
Environment -- 训练的基础信息，包括随机数种子、训练的名字（标识）
"""
env_conf = dict(
    # 随机数种子
    seed=42,

    # 本次训练的命名
    name='default_train',

    # 其他
    n_gpus=1,
    accelerator='gpu',
    strategy='ddp'
)


"""
Train -- 训练流程的参数，指定了Trainer的行为
"""
train_conf = dict(
    # 基础训练参数
    learning_rate=3e-4,
    weight_decay=0.1,
    adam_epsilon=1e-3,
    warmup_steps=0,
    max_epochs=5,
    accumulate_grad_batches=2,

    # batch参数
    train_batch_size=4,
    eval_batch_size=16
)


"""
Model -- 模型参数，指定了Model、FineTuner的行为
"""
model_conf = dict(

)

plm_model_conf = dict(
    model_name='bert-base-chinese',
    max_seq_length=512,
)

prompt_model_conf = dict(
    prompt_type='',
    # 是否将prompt堆叠来构造训练数据
    compact=False
)

extract_model_conf = dict(
    class_cnt=len(duie_relations)
)
"""
model
    - plm_model
        - prompt_model
        - extract_model
"""



"""
Logger -- 指定了Logger的行为
"""
logger_conf = dict(
    logger_dir='tb_log/',
    every_n_epochs=1,
)

ckp_conf = dict(
    save_top_k=-1,
    monitor='val_loss',
    dirpath='t5-checkpoints/',
)