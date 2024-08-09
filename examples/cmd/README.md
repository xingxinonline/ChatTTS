#[oral_(0-9)]: 控制口音强度

#[laugh_(0-2)]: 控制笑声

#[break_(0-7)]: 控制停顿时间

params_refine_text=ChatTTS.Chat.RefineTextParams(
            prompt='[oral_0][laugh_0][break_5]',
        )

#[speed_(0-9)]: 控制语速

#temperature ：0.00001-1.0 控制音频情感波动性，范围为 0-1，数字越大，波动性越大

#top_P ：控制音频的情感相关性，范围为 0.1-0.9，数字越大，相关性越高

#top_K ：控制音频的情感相似性，范围为 1-20，数字越小，相似性越高

params_infer_code=ChatTTS.Chat.InferCodeParams(
            prompt='[speed_9]',
            spk_emb=spk_emb_str,
            temperature=0.0003,  # using custom temperature
            top_P=0.7,  # top P decode
            top_K=20,  # top K decode


python examples/cmd/kws_src.py "机械臂归零[lbreak]" "机械臂前进[lbreak]" "机械臂后退[lbreak]" "机械臂左转[lbreak]" "机械臂右转[lbreak]" "机械臂上升[lbreak]" "机械臂下降[lbreak]" "机械臂抓取[lbreak]" "机械臂松开[lbreak]"