from cores import Preprocessor
from typing import List, Union
from datasets import Dataset
import re
from EmailParser.EmailParser import EmailParser


@Preprocessor.register("email-body")
class EmailBodyPreprocessor(Preprocessor):

    def __init__(self) -> None:
        pass

    @Preprocessor.cache()
    def __call__(self, dataset: Dataset) -> Dataset:

        def dropempty(sample):
            for key in sample.keys():
                if sample[key] == '':
                    return False
            return True

        dataset = dataset.filter(dropempty)
        dataset = dataset.map(lambda sample: {'body': self._prepare(
            sample['body'], sample['subject'])}, batched=True, num_proc=self._parent.num_workers)
        dataset = dataset.map(lambda sample: {'body': re.sub(
            '\s+\S*$', '', re.sub('^ ', '', re.sub(' +', ' ', re.sub('(.*?(\\xa0)|(\\n).*?)', '', sample['body']))))})

        return dataset

    def _prepare(self,
                 input: Union[str, List[str]],
                 subject: Union[None, str, List[str]] = None) -> List[str]:

        ret: List[str] = []

        if isinstance(input, List):
            for batch_sample in input:
                batch_sample_output = self._prepare_atomic(
                    batch_sample, subject)
                ret.append(batch_sample_output)
        else:
            ret.append(self._prepare_atomic(input, subject))

        return ret

    def _prepare_atomic(self,
                        input: str,
                        subject: Union[None, str] = None) -> str:

        body = self._normalize_body(input)
        text_to_analyze = ''
        if self._check_subject_is_fwd(subject) or self._check_subject_is_reply(subject):
            paragraphs = body.split('\n\n')
            for paragraph in paragraphs:
                if len(paragraph) > 3:
                    if len(paragraph) > 20:
                        normalized_paragraph = self._check_parapgraph_is_reply_header(
                            paragraph.replace('= ', ''))
                        if normalized_paragraph:
                            break
                        else:
                            text_to_analyze += '\n\n' + paragraph
                    else:
                        text_to_analyze += '\n\n' + paragraph
        else:
            text_to_analyze = body

        try:
            ep = EmailParser()
            croped_text = ep.remove_signature(text_to_analyze.strip())
            # print('-----signature----')
            # print(text_to_analyze.replace(croped_text, ''))
            return croped_text
        except Exception as e:
            # tb = traceback.format_exc()
            # logger.error('error code {}, error traceback {}'.format(str(e), str(tb)))
            # print('error signature')
            return text_to_analyze

    def _normalize_body(self, body):
        cropped_body = body.replace('=\n', '').replace(
            '= ', '').replace('\r\n', '\n')
        lines = re.sub(r"\n\s+\n", "\n\n", cropped_body).split('\n')
        normalized_body = ''
        for line in lines:
            normalized_line = line.lstrip('>').lstrip().lstrip(
                '<br>').replace('=2E', '').replace('= ', '')
            normalized_body += '\n' + normalized_line

        return normalized_body

    def _check_subject_is_fwd(self, subject):
        is_fwd = False
        if subject.lower().startswith('fw ') or subject.lower().startswith('fwd ') \
                or subject.lower().startswith('fw:') or subject.lower().startswith('fwd:'):
            is_fwd = True
        return is_fwd

    def _check_subject_is_reply(self, subject):
        is_reply = False
        if subject.lower().startswith('re ') or subject.lower().startswith('re:'):
            is_reply = True
        return is_reply

    def _parse_addresses(self, string):
        regex = re.compile(("([a-z0-9!#$%&'*+\/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+\/=?^_`"
                            "{|}~-]+)*(@|\sat\s)(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?(\.|"
                            "\sdot\s))+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?)"))
        mailboxes = []
        lst = (email[0] for email in re.findall(regex, string))
        for item in lst:
            mailboxes.append(item)
        return mailboxes

    def _is_date(self, string, fuzzy=False):
        try:
            self.parse(string, fuzzy=fuzzy)
            return True
        except ValueError:
            return False

    def _check_parapgraph_is_reply_header(self, paragraph):
        lines = paragraph.split('\n')
        match_count = 0
        normalized_paragraph = ''
        other_match_count = 0
        for line in lines:
            normalized_line = line.replace('*', '').lstrip().rstrip().lower()
            normalized_paragraph += '\n' + normalized_line
            first_type_count = 0
            second_type_count = 0
            if len(self._parse_addresses(line)):
                first_type_count += 1
            try:
                if self._is_date(line, True):
                    first_type_count += 1
            except Exception as e:
                pass
            if normalized_line.startswith('on'):
                first_type_count += 1
            if normalized_line.endswith('wrote:'):
                first_type_count += 1
                if other_match_count == 1:
                    return paragraph

            if '--- on' in normalized_line:
                second_type_count += 1
            if 'wrote ---' in normalized_line:
                second_type_count += 1

            if second_type_count == 2 or first_type_count >= 3:
                return normalized_line

            if first_type_count == 2:
                other_match_count = 1

            if normalized_line.startswith('from:'):
                match_count += 1
            elif normalized_line.startswith('to:'):
                match_count += 1
            elif normalized_line.startswith('subject:'):
                match_count += 1
            elif normalized_line.startswith('date:') or normalized_line.startswith('sent:'):
                match_count += 1
            elif normalized_line.startswith('cc:'):
                match_count += 1
        if match_count >= 3:
            return normalized_paragraph
        else:
            return None
