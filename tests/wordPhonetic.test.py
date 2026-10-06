"""Exact CTC semantics and whole-word boundaries against enumerated paths."""
from pathlib import Path
import itertools
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.runtime/wake-validation'),str(ROOT/'word_training')]
import numpy as np
from phonetic import keyword_probability,targets,forced_alignment,posterior


def hard(sequence):return np.eye(6)[sequence]


class PhoneticChecks(unittest.TestCase):
    def test_combined_onnx_uses_one_input_and_matches_blended_probabilities(self):
        import onnx
        import onnxruntime as ort
        from onnx import helper,TensorProto
        from choose import merge,blend
        graph=helper.make_graph([helper.make_node('Sigmoid',['log_mel'],['probability'])],'binary',
            [helper.make_tensor_value_info('log_mel',TensorProto.FLOAT,['batch'])],
            [helper.make_tensor_value_info('probability',TensorProto.FLOAT,['batch'])])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);model.ir_version=8
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as directory:
            source=Path(directory)/'binary.onnx';destination=Path(directory)/'combined.onnx'
            onnx.save(model,source);merge(source,source,destination,.25)
            options=ort.SessionOptions();options.intra_op_num_threads=2
            session=ort.InferenceSession(str(destination),options,providers=['CPUExecutionProvider'])
            self.assertEqual([i.name for i in session.get_inputs()],['log_mel'])
            for batch in [1,4]:
                values=np.linspace(-4,4,batch).astype(np.float32)
                probability=1/(1+np.exp(-values))
                actual=session.run(['probability'],{'log_mel':values})[0]
                np.testing.assert_allclose(actual,blend(probability,probability,.25),atol=1e-7,rtol=1e-7)
    def test_whole_word_boundaries_and_ctc_repeats(self):
        for sequence in [[1,2,3],[0,1,1,0,2,2,0,3,0],[5,4,1,2,3,4,5],[1,2,3,4,1,2,3]]:
            self.assertEqual(keyword_probability(hard(sequence)),1)
        for sequence in [[5,2,1,2,3],[1,2,3,5],[5,2,3],[1,0,1,2,3],[0,0,0]]:
            self.assertEqual(keyword_probability(hard(sequence)),0)
    def test_probability_matches_exhaustive_ctc_paths(self):
        rng=np.random.default_rng(42);prob=rng.dirichlet(np.ones(6),5)
        expected=0.
        for path in itertools.product(range(6),repeat=5):
            collapsed=[label for i,label in enumerate(path) if label and (i==0 or label!=path[i-1])]
            words=[];word=[]
            for label in collapsed+[4]:
                if label==4:words.append(word);word=[]
                else:word.append(label)
            if [1,2,3] in words:expected+=np.prod([prob[i,label] for i,label in enumerate(path)])
        self.assertAlmostEqual(keyword_probability(prob),expected,places=12)
    def test_text_mapping_preserves_word_boundaries(self):
        self.assertEqual(targets('БОТ, включи!'),[1,2,3,4,5])
        self.assertEqual(targets('робот'),[5,2,1,2,3])
        self.assertEqual(targets('боту'),[1,2,3,5])
    def test_invalid_probabilities_rejected(self):
        with self.assertRaises(ValueError):keyword_probability(np.zeros((4,6)))
        with self.assertRaises(ValueError):keyword_probability(np.full((4,6),np.nan))
    def test_alignment_uses_annotation_even_when_teacher_mislabels_initial_sound(self):
        probability=np.full((7,6),.001)
        for time,label in enumerate([0,5,0,2,0,3,0]):probability[time,label]=.99
        probability[1,1]=.008
        alignment=forced_alignment(probability,[1,2,3])
        collapsed=[label for i,label in enumerate(alignment) if label and (not i or label!=alignment[i-1])]
        self.assertEqual(collapsed,[1,2,3])
    def test_exported_onnx_word_search_matches_reference_for_dynamic_batches(self):
        import onnx
        import onnxruntime as ort
        from onnx import helper,TensorProto
        from phonetic_onnx import add_keyword_output
        graph=helper.make_graph([helper.make_node('Identity',['log_mel'],['logits'])],'test',
            [helper.make_tensor_value_info('log_mel',TensorProto.FLOAT,['batch','time',6])],
            [helper.make_tensor_value_info('logits',TensorProto.FLOAT,['batch','time',6])])
        model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);model.ir_version=8
        with tempfile.TemporaryDirectory(dir=ROOT/'.runtime') as directory:
            source=Path(directory)/'source.onnx';destination=Path(directory)/'word.onnx'
            onnx.save(model,source);add_keyword_output(source,destination,.7)
            options=ort.SessionOptions();options.intra_op_num_threads=2
            session=ort.InferenceSession(str(destination),options,providers=['CPUExecutionProvider'])
            for batch in [1,4]:
                logits=np.random.default_rng(batch).normal(size=(batch,5,6)).astype(np.float32)
                expected=keyword_probability(posterior(logits,.7))
                actual=session.run(['probability'],{'log_mel':logits})[0]
                np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10)


if __name__=='__main__':unittest.main()
