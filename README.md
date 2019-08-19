# bert-multi-gpu

Feel free to fine tune large BERT models with large batch size easily. Multi-GPU and FP16 are supported.

## Dependencies

- Tensorflow
  - tensorflow >= 1.11.0   # CPU Version of TensorFlow.
  - tensorflow-gpu  >= 1.11.0  # GPU version of TensorFlow.
- NVIDIA Collective Communications Library (NCCL)



## Features

- CPU/GPU/TPU Support
- **Multi-GPU Support**: [`tf.distribute.MirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy) is used to achieve Multi-GPU support for this project, which mirrors vars to distribute across multiple devices and machines. The maximum batch_size for each GPU is almost the same as [bert](https://github.com/google-research/bert/blob/master/README.md#out-of-memory-issues). So **global batch_size** depends on how many GPUs there are.
- **FP16 Support**: [FP16](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) allows you to use a larger batch_size. And training speed will increase by 70~100% on Volta GPUs, but may be slower on Pascal GPUs([REF1](https://github.com/tensorflow/tensorflow/issues/15585#issuecomment-361769151), [REF2](https://github.com/HaoyuHu/bert-multi-gpu/issues/1#issuecomment-493363383)).
- **SavedModel Export**



## Usage

### Run Classifier

List some optional parameters below:

- `task_name`: The name of task which you want to fine tune, you can define your own task by implementing `DataProcessor` class.
- `do_lower_case`: Whether to lower case the input text. Should be True for uncased models and False for cased models. Default value is `true`.
- `do_train`: Fine tune classifier or not. Default value is `false`.
- `do_eval`: Evaluate classifier or not. Default value is `false`.
- `do_predict`: Predict by classifier recovered from checkpoint or not. Default value is `false`.
- `save_for_serving`: Output SavedModel for tensorflow serving. Default value is `false`.
- `data_dir`: Your original input data directory.
- `vocab_file`, `bert_config_file`, `init_checkpoint`: Files in BERT model directory.
- `max_seq_length`: The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded. Default value is `128`.
- `train_batch_size`: Batch size for [**each GPU**](<https://stackoverflow.com/questions/54327610/does-tensorflow-estimator-take-different-batches-for-workers-when-mirroredstrate/54332773#54332773>). For example, if `train_batch_size` is 16, and `num_gpu_cores` is 4, your **GLOBAL** batch size is 16 * 4 = 64.
- `learning_rate`: Learning rate for Adam optimizer initialization.
- `num_train_epochs`: Train epoch number.
- `use_gpu`: Use GPU or not.
- `num_gpu_cores`: Total number of GPU cores to use, only used if `use_gpu` is True.
- `use_fp16`: Use [`FP16`](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) or not.
- `output_dir`: **Checkpoints** and **SavedModel(.pb) files** will be saved in this directory.

```shell
python run_custom_classifier.py \
  --task_name=QQP \
  --do_lower_case=true \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --save_for_serving=true \
  --data_dir=/cfs/data/glue/QQP \
  --vocab_file=/cfs/models/bert-large-uncased/vocab.txt \
  --bert_config_file=/cfs/models/bert-large-uncased/bert_config.json \
  --init_checkpoint=/cfs/models/bert-large-uncased/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --use_gpu=true \
  --num_gpu_cores=4 \
  --use_fp16=true \
  --output_dir=/cfs/outputs/bert-large-uncased-qqp
```



### Run Sequence Labeling

List some optional parameters below:

- `task_name`: The name of task which you want to fine tune, you can define your own task by implementing `DataProcessor` class.
- `do_lower_case`: Whether to lower case the input text. Should be True for uncased models and False for cased models. Default value is `true`.
- `do_train`: Fine tune model or not. Default value is `false`.
- `do_eval`: Evaluate model or not. Default value is `false`.
- `do_predict`: Predict by model recovered from checkpoint or not. Default value is `false`.
- `save_for_serving`: Output SavedModel for tensorflow serving. Default value is `false`.
- `data_dir`: Your original input data directory.
- `vocab_file`, `bert_config_file`, `init_checkpoint`: Files in BERT model directory.
- `max_seq_length`: The maximum total input sequence length after WordPiece tokenization. Sequences longer than this will be truncated, and sequences shorter than this will be padded. Default value is `128`.
- `train_batch_size`: Batch size for [**each GPU**](<https://stackoverflow.com/questions/54327610/does-tensorflow-estimator-take-different-batches-for-workers-when-mirroredstrate/54332773#54332773>). For example, if `train_batch_size` is 16, and `num_gpu_cores` is 4, your **GLOBAL** batch size is 16 * 4 = 64.
- `learning_rate`: Learning rate for Adam optimizer initialization.
- `num_train_epochs`: Train epoch number.
- `use_gpu`: Use GPU or not.
- `num_gpu_cores`: Total number of GPU cores to use, only used if `use_gpu` is True.
- `use_fp16`: Use [`FP16`](https://en.wikipedia.org/wiki/Half-precision_floating-point_format) or not.
- `output_dir`: **Checkpoints** and **SavedModel(.pb) files** will be saved in this directory.

```shell
python run_seq_labeling.py \
  --task_name=PUNCT \
  --do_lower_case=true \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --save_for_serving=true \
  --data_dir=/cfs/data/PUNCT \
  --vocab_file=/cfs/models/bert-large-uncased/vocab.txt \
  --bert_config_file=/cfs/models/bert-large-uncased/bert_config.json \
  --init_checkpoint=/cfs/models/bert-large-uncased/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=10.0 \
  --use_gpu=true \
  --num_gpu_cores=4 \
  --use_fp16=true \
  --output_dir=/cfs/outputs/bert-large-uncased-punct
```



## What's More

### Add custom task

You can define your own task data processor by implementing `DataProcessor` class. 

Then, add your `CustomProcessor` to processors.

Finally, you can pass `--task=your_task_name` to python script. 

```python
# Create custom task data processor in run_custom_classifier.py
class CustomProcessor(DataProcessor):
    """Processor for the Custom data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(read_custom_train_lines(data_dir), 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(read_custom_dev_lines(data_dir), 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(read_custom_test_lines(data_dir), 'test')

    def get_labels(self):
        """See base class."""
        return your_label_list # ["label-1", "label-2", "label-3", ..., "label-k"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training/evaluation/testing sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # text_b can be None
            (guid, text_a, text_b, label) = parse_your_data_line(line)
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

# Add CustomProcessor to processors in run_custom_classifier.py
def main(_):
    # ...
    # Register 'custom' processor name to processors, and you can pass --task_name=custom to this script
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "xnli": XnliProcessor,
        "qqp": QqpProcessor,
        "custom": CustomProcessor,
    }
    # ...
```



### Tensorflow serving

If `--save_for_serving=true` is passed to `run_custom_classifier.py` or `run_seq_labeling.py`, python script will export **SavedModel** file to `output_dir`. Now you are good to go.

- Install the [SavedModel CLI](https://www.tensorflow.org/guide/saved_model#install_the_savedmodel_cli) by installing a pre-built Tensorflow binary(usually already installed on your system at pathname `bin\saved_model_cli`) or building TensorFlow from source code.

- Check your **SavedModel** file:

  ```shell
  saved_model_cli show --dir <bert_savedmodel_output_path>/<timestamp> --all
  
  # For example:
  saved_model_cli show --dir tf_serving/bert_base_uncased_multi_gpu_qqp/1557722227/ --all
  
  # Output:
  # signature_def['serving_default']:
  #   The given SavedModel SignatureDef contains the following input(s):
  #     inputs['input_ids'] tensor_info:
  #         dtype: DT_INT32
  #         shape: (-1, 128)
  #         name: input_ids:0
  #     inputs['input_mask'] tensor_info:
  #         dtype: DT_INT32
  #         shape: (-1, 128)
  #         name: input_mask:0
  #     inputs['label_ids'] tensor_info:
  #         dtype: DT_INT32
  #         shape: (-1)
  #         name: label_ids:0
  #     inputs['segment_ids'] tensor_info:
  #         dtype: DT_INT32
  #         shape: (-1, 128)
  #         name: segment_ids:0
  #   The given SavedModel SignatureDef contains the following output(s):
  #     outputs['probabilities'] tensor_info:
  #         dtype: DT_FLOAT
  #         shape: (-1, 2)
  #         name: loss/Softmax:0
  #   Method name is: tensorflow/serving/predict
  ```

- Install [Bazel](https://docs.bazel.build/versions/master/install.html) and compile **tensorflow_model_server**.

  ```shell
  cd /your/path/to/tensorflow/serving
  bazel build -c opt //tensorflow_serving/model_servers:tensorflow_model_server
  ```

- Start tensorflow serving to listen on port for **HTTP/REST API** or **gRPC API**, `tensorflow_model_server` will initialize the models in `<bert_savedmodel_output_path>`.

  ```shell
  # HTTP/REST API
  bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --rest_api_port=<rest_api_port> --model_name=<model_name> --model_base_path=<bert_savedmodel_output_path>
  
  # For example:
  bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --rest_api_port=9000 --model_name=bert_base_uncased_qqp --model_base_path=/root/tf_serving/bert_base_uncased_multi_gpu_qqp --enable_batching=true
  
  # Output:
  # 2019-05-14 23:26:38.135575: I tensorflow_serving/core/loader_harness.cc:86] Successfully loaded servable version {name: bert_base_uncased_qqp version: 1557722227}
  # 2019-05-14 23:26:38.158674: I tensorflow_serving/model_servers/server.cc:324] Running gRPC ModelServer at 0.0.0.0:8500 ...
  # 2019-05-14 23:26:38.179164: I tensorflow_serving/model_servers/server.cc:344] Exporting HTTP/REST API at:localhost:9000 ...
  ```

- Make a request to test your latest serving model.

  ```shell
  curl -H "Content-type: application/json" -X POST -d '{"instances": [{"input_ids": [101,2054,2064,2028,2079,2044,16914,5910,1029,102,2054,2079,1045,2079,2044,2026,16914,5910,1029,102,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "input_mask": [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "segment_ids": [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], "label_ids":[0]}]}'  "http://localhost:9000/v1/models/bert_base_uncased_qqp:predict"
  
  # Output:
  # {"predictions": [[0.608512461, 0.391487628]]}
  ```


## Stargazers over time
[![Stargazers over time](https://starchart.cc/HaoyuHu/bert-multi-gpu.svg)](https://starchart.cc/HaoyuHu/bert-multi-gpu)
  

## License

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright [yyyy] [name of copyright owner]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

```
