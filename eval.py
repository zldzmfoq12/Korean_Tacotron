import argparse
import os
import re
from hparams import hparams, hparams_debug_string
from synthesizer import Synthesizer


sentences = [
  # 장기하와 얼굴들 ㅋ 가사:
  '나는 대체로 선생님의 말씀에 귀를 기울이지 않았다|나는 데체로 선셍니미 말쓰메 주이를 기우리지 아낟따.',
  '나는 눈썹은 까만데 머리는 갈색이요|저는 눈써븐 까만데, 머리는 갈세기에요.',
  '제가 처음 남편을 만났을 때 남편은 대머리가 아니었어요|제가 처음 남펴늘 만나쓸 떼, 남펴는 데머리가 아니어써요.'
  # 장기하와 얼굴들 새해복 가사:
#   '간장 공장 공장장',
#   '한양양장점 옆 한양양장점',
#   '후회한 시간을 후회할 거잖아',
]


def get_output_base_path(checkpoint_path):
    base_dir = os.path.dirname(checkpoint_path)
    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(checkpoint_path)
    name = 'eval-concat-%d' % int(m.group(1)) if m else 'eval'
    return os.path.join(base_dir, name)


def run_eval(args):
    print(hparams_debug_string())
    synth = Synthesizer()
    synth.load(args.checkpoint)
    base_path = get_output_base_path(args.checkpoint)
    # sentences=[]
    # with open('./eval_concat.txt', encoding='utf-8') as f:
    #     for line in f:
    #         try:
    #             parts = line.strip().replace('"', '').split('|')
    #             c_text = parts[3]
    #             p_text = parts[4]
    #             sentences.append(c_text+'|'+p_text)
    #         except:
    #             pass
    for i, text in enumerate(sentences):
        path = '%s-%d.wav' % (base_path, i)
        print('Synthesizing: %s' % path)
        with open(path, 'wb') as f:
            f.write(synth.synthesize(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--gpu', default='1')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    hparams.parse(args.hparams)
    run_eval(args)


if __name__ == '__main__':
  main()
