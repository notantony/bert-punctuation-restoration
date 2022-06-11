from pathlib import Path
from typing import Union, List

from lxml import etree

from utils.paths import DEV_XML, TEST_XML, TRAIN_DATA


def load_train_data(train_data: Union[Path, str] = TRAIN_DATA) -> List[str]:
    print(f'Loading train data from: `{train_data}`')
    with open(train_data, 'r', encoding='UTF8') as ds:
        lines = [l for l in ds.readlines() if not l.startswith('<')]
    print(f'Loaded {len(lines)} lines')
    return lines


def load_test_data(test_xml: Union[Path, str] = TEST_XML) -> List[str]:
    print(f'Loading data from: `{test_xml}`')
    lines = []
    with open(test_xml, 'r', encoding='UTF8') as ds:
        text = ''.join(ds.readlines()[1:])
        data = etree.HTML(text)
        r = data.xpath('//seg')
        for seg in r:
            if not seg.text.isspace():
                lines.append(seg.text)
    print(f'Loaded {len(lines)} lines')
    return lines


def load_dev_data(dev_xml: Union[Path, str] = DEV_XML) -> List[str]:
    return load_test_data(dev_xml)
