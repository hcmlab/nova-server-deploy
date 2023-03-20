# Copyright (c) 2023, Tobias Hallmen

import sys
if not hasattr(sys, 'argv'):
    sys.argv  = ['']


#from pathlib import Path
#from hcai_datasets.hcai_nova_dynamic.hcai_nova_dynamic_iterable import HcaiNovaDynamicIterable
#from hcai_dataset_utils.bridge_pytorch import BridgePyTorch
#from torchaudio import transforms as audio_transforms
#from torchvision import transforms as vision_transforms
import torch
#import time
#import whisper
import numpy as np
import logging

from lhotse import Recording, RecordingSet, CutSet, align_with_torchaudio, annotate_with_whisper

class TrainerClass:
    def __init__(self, ds_iter, logger, request_form=None):
        self.model = None
        self.ds_iter = ds_iter
        self.logger = logger
        self.data = None
        self.predictions = None
        self.DEPENDENCIES = []
        self.OPTIONS = {'loudness normalisation': False, 'whisper': 'base', 'translate': False, 'sentence logic': False, 'align': True, 'wav2vec2': 'base', 'diarise': False, 'roles': None, 'split audio': False}
        self.request_form = request_form
        self.delete = []

    def train(self):
        pass
        
    def save(self, path):
        pass
    
    def load(self, path):
        pass

    def preprocess(self):
        audios = [f"{self.ds_iter.nova_data_dir}/{self.ds_iter.dataset}/{sess}/{rol}.{strim}.{self.ds_iter.data_info[f'{rol}.{strim}'].file_ext}"
                  for strim in self.ds_iter.data_streams for rol in self.ds_iter.roles for sess in self.ds_iter.sessions]
        self.logger.info('Found:')
        for a in audios:
            self.logger.info(a.split('/')[-2] + '/' + a.split('/')[-1])
        if self.OPTIONS['loudness normalisation']:
            import soundfile
            import pyloudnorm
            import tempfile
            records = []
            for a in audios:
                data, sr = soundfile.read(a)
                data = pyloudnorm.normalize.peak(data, -1.0)
                tmpf = tempfile.NamedTemporaryFile(delete=False)
                self.delete.append(tmpf.name)
                soundfile.write(tmpf, data, sr, format='wav')
                tmpf.close()  # file has to be closed before opening again on windows
                records.append(Recording.from_file(tmpf.name, a.split('/')[-1][:-4]))  # remove '.wav' at the end
            self.data = RecordingSet.from_recordings(records)
        else:
            self.data = RecordingSet.from_recordings(Recording.from_file(a) for a in audios)
    
    def postprocess(self):
        import os
        for x in self.delete:
            os.unlink(x)
        result = self.predictions
        result['values'] = [] 
        result['confidences'] = []   
        return result


    def predict(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.cuda.empty_cache()

        #cuts = annotate_with_whisper(self.data, model_name=self.request_form['options']['whisper'])
        #cuts_aligned = align_with_torchaudio(cuts, bundle_name=f"WAV2VEC2_ASR_{self.request_form['options']['wav2vec2'].upper()}_960H")

        # list() to yield; empty_cache to release VRAM, lhotse doesn't do this
        if self.OPTIONS['translate']:
            self.logger.info('translating')
            cuts = list(annotate_with_whisper(self.data, model_name=self.OPTIONS['whisper'], device=device, task='translate'))
        else:
            self.logger.info('transcribing')
            cuts = list(annotate_with_whisper(self.data, model_name=self.OPTIONS['whisper'], device=device))
        torch.cuda.empty_cache()

        if self.OPTIONS['align']:
            self.logger.info('aligning')
            cuts = list(align_with_torchaudio(cuts, bundle_name=f"WAV2VEC2_ASR_{self.OPTIONS['wav2vec2'].upper()}_960H",device=device))
            torch.cuda.empty_cache()

        writer = CutSet.open_writer(None)
        for cut in cuts:
            writer.write(cut)

        transcript_dict = {}
        for cut in writer.items:
            sentences = []
            if self.OPTIONS['sentence logic']:
                self.logger.info('applying sentence logic')
                begin = -1
                end = -1
                flattened = {'text': '', 'alignment': []}
                for entry in cut.supervisions:
                    if entry.alignment is not None:
                        if len(flattened['text']) == 0:
                            flattened['text'] = entry.text
                        else:
                            flattened['text'] += ' ' + entry.text
                        flattened['alignment'].extend(entry.alignment['word'])
                    else:
                        continue

                text = flattened['text'].split()
                mean = [0, 0]
                for i, word in enumerate(text):
                    mean[0] += 1
                    #mean[1] += 1 / flattened['alignment'][i][3]  # harmonic mean
                    mean[1] += flattened['alignment'][i][3]  # arithmetic mean
                    if begin == -1:
                        begin = i
                    # maybe define a complement set of rules for sentence ends
                    if end == -1 and (word[-1] not in ('.', '?', '!') or word[-3:] in ('Dr.')):#, '...')):
                        continue
                    else:
                        end = i
                        sentences.append({'text': ' '.join(text[begin:end + 1]), 'start': flattened['alignment'][begin][1],
                                          'stop': flattened['alignment'][end][1] + flattened['alignment'][end][2],
                                          #'conf': mean[0] / mean[1]})  # harmonic mean
                                          'conf': mean[1] / mean[0]})  # arithmetic mean
                        begin = -1
                        end = -1
                        mean = [0, 0]
            else:
                for sup in cut.supervisions:
                    if sup.alignment:
                        sentences.append({'text': sup.text, 'start': sup.alignment['word'][0].start,
                                          'stop': sup.alignment['word'][-1].start+sup.alignment['word'][-1].duration,
                                          'conf': sum(l := [x.score for x in sup.alignment['word']])/len(l)})
                    else:
                        sentences.append({'text': sup.text, 'start': sup.start,
                                          'stop': sup.end,
                                          'conf': 0})

            tmp = {f"{sent['start']}_{sent['stop']}": {cut.id: {'name': sent['text'], 'conf': sent['conf']}} for sent in sentences}
            if not transcript_dict:
                transcript_dict = tmp
            else:
                if not (intersect := list(filter(transcript_dict.__contains__, tmp.keys()))):
                    transcript_dict |= tmp
                else:
                    transcript_dict = tmp | transcript_dict
                    for k in intersect:
                        transcript_dict[k] |= tmp[k]

        if self.OPTIONS['diarise']:
            self.logger.info('diarising')
            torch.cuda.empty_cache()
            from sklearn.cluster import AgglomerativeClustering
            from pyannote.audio import Audio
            from pyannote.core import Segment
            from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
            from pyannote.audio.pipelines.clustering import AgglomerativeClustering as pyannAC
            # valid other models
            # get_embedding = PretrainedSpeakerEmbedding("pyannote/embedding")
            # get_embedding = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb")
            # get_embedding = PretrainedSpeakerEmbedding("nvidia/speakerverification_en_titanet_large")
            spkr_embed_model = PretrainedSpeakerEmbedding('speechbrain/spkrec-ecapa-voxceleb', device=device)
            roles = self.OPTIONS['roles'].split(',')
            audio = Audio()
            audio_file = self.data.data[cut.id].sources[0].source
            audio_duration = audio.get_duration(audio_file)
            spkr_embeds = np.zeros((len(sentences), 192))
            for i, sent in enumerate(sentences):
                # assuming it is only one file/path in the CutSet
                data, sr = audio.crop(audio_file, Segment(sent['start'], min(sent['stop'], audio_duration)))
                spkr_embeds[i] = spkr_embed_model(data[np.newaxis])  # add batch axis

            # sklearn: pyannote uses cosine metric, but then "warding" doesn't work in sklearn
            #spkr_cluster = AgglomerativeClustering(len(roles)).fit(spkr_embeds)
            ##spkr_cluster = AgglomerativeClustering(len(roles), linkage='complete').fit(spkr_embeds)
            #spkr_label_to_role = {l: r for l, r in zip(set(spkr_cluster.labels_), roles)}

            #for i, (k, v) in enumerate(transcript_dict.items()):
            #    transcript_dict[k] |= {spkr_label_to_role[spkr_cluster.labels_[i]]+'.audio': transcript_dict[k][cut.id]}


            # pyannote probably needs its own segmentation beforehand
            #spkr_cluster2 = pyannAC(metric='cosine')
            #spkr_cluster2.instantiate({"method": "average", "threshold": 1.0, "min_cluster_size": 1})
            #cluster, _ = spkr_cluster2(spkr_embeds, min_clusters=1, max_clusters=np.inf, num_clusters=len(roles))

            # finch
            from finch import FINCH
            cluster_partition, n_part_clust, part_labels = FINCH(spkr_embeds, req_clust=len(roles))
            spkr_label_to_role = {l: r for l, r in zip(set(part_labels), roles)}

            for i, (k, v) in enumerate(transcript_dict.items()):
                transcript_dict[k] |= {spkr_label_to_role[part_labels[i]]+'.audio': transcript_dict[k][cut.id]}

            if self.OPTIONS['split audio']:
                self.logger.info('splitting audio')
                import soundfile as sf
                data, sr = sf.read(audio_file)
                for label in set(part_labels):
                    data_cpy = data.copy()
                    start = 0
                    for i, sent in enumerate(sentences):
                        if label == part_labels[i]:
                            data_cpy[int(start*sr):int(sent['start']*sr)] = 0
                            start = sent['stop']
                        else:
                            continue
                    data_cpy[int(start*sr):] = 0
                    sf.write(audio_file.replace(cut.id, spkr_label_to_role[label]+'.audio'), data_cpy, sr)


        torch.cuda.empty_cache()
        self.predictions = transcript_dict
