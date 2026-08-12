# WANDS 数据下载与治理说明

## 1. 数据源

- 数据集：Wayfair ANnotation Dataset（WANDS）
- 官方仓库：[wayfair/WANDS](https://github.com/wayfair/WANDS)
- 固定提交：[3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5](https://github.com/wayfair/WANDS/tree/3b74dcf4ba29ab8ff3e6a50b5b09fc627cb882b5)
- 许可证：MIT
- 论文用途：商品搜索相关性评测

项目固定提交而不是直接跟随 `main`，保证不同机器和不同时间得到相同文件。

## 2. 为什么第一版选择 WANDS

WANDS 同时具备：

- 真实家具商品目录。
- 真实商品搜索查询。
- 人工标注的查询—商品相关性。
- 商品描述、结构化特征、类别和评分统计。
- 允许本地复现的公开许可证。

它既能支撑用户可理解的商品搜索 Web UI，也能用人工标注客观评测检索质量。

## 3. 本地需要下载的文件

| 文件 | 官方大小 | 本地用途 | 是否提交 Git |
| --- | ---: | --- | --- |
| `product.csv` | 90,621,131 字节 | 商品存储、索引和回答证据 | 否 |
| `query.csv` | 19,942 字节 | 评测查询 | 否 |
| `label.csv` | 5,736,234 字节 | 人工相关性标签 | 否 |
| `LICENSE` | 1,065 字节 | 许可证留档 | 否，随数据下载 |
| `README.md` | 3,747 字节 | 官方字段说明 | 否，随数据下载 |
| `Product Search Relevance Annotation Guidelines.pdf` | 15,769,412 字节 | 标注规则核验 | 否，随数据下载 |

下载后位置：

```text
data/raw/wands/
├── LICENSE
├── Product Search Relevance Annotation Guidelines.pdf
├── README.md
├── label.csv
├── product.csv
└── query.csv
```

## 4. 下载命令

```bash
bash scripts/download_wands.sh
```

脚本会：

1. 从固定 Git 提交下载到临时目录。
2. 验证六个文件的 SHA-256。
3. 验证三个数据文件的行数。
4. 验证 TSV 表头。
5. 所有验证通过后才替换 `data/raw/wands/` 中的对应文件。

原始文件被 `.gitignore` 排除，仓库只保存下载脚本和校验清单。

## 5. 重要格式说明

三个数据文件虽然扩展名为 `.csv`，实际使用 **Tab（制表符）分隔**。

Python 标准库示例：

```python
import csv

with open("data/raw/wands/product.csv", encoding="utf-8") as source:
    rows = csv.DictReader(source, delimiter="\t")
```

如果按逗号解析，整行会被错误识别成一个字段。导入器必须对此编写回归测试。

## 6. 已核验数据量

| 数据 | 数据行数 | 包含表头的文件行数 |
| --- | ---: | ---: |
| 商品 | 42,994 | 42,995 |
| 查询 | 480 | 481 |
| 相关性标注 | 233,448 | 233,449 |

标注分布：

| 标签 | 数量 |
| --- | ---: |
| Exact | 25,614 |
| Partial | 146,633 |
| Irrelevant | 61,201 |

## 7. SHA-256

权威校验文件为 [`data/WANDS_SHA256SUMS`](../data/WANDS_SHA256SUMS)。当前值：

```text
d993926254572e6eba96c8fd87cc549a17fb91ad3748308036eee4cf92b10ac6  product.csv
63b61660560fecc33ec490804c7e2b81402ee3e7c31a9cbb5e03736639f68e95  query.csv
c11fe81ad62f17f56f316b0ec9630ebe8fbe1393578cb0ca4f05c17253a180ef  label.csv
e3ce14610132897db9f64e21d7871a7a60c0bc04364ec61e4faa99643c5072d6  LICENSE
a967f26f3d97102baa38ece95c08c0625fedf0d65bfc7cccaed7ec26f739c242  README.md
35d2923435f3f63ca450e73666a594cd265ab42e0810b25556d91c093575c625  Product Search Relevance Annotation Guidelines.pdf
```

## 8. 数据质量事实

本次下载的商品数据中：

- 商品名称非空：42,994（100%）。
- 商品特征非空：42,994（100%）。
- 商品描述非空：36,986（约 86%）。
- 平均评分非空：33,542（约 78%）。
- 评论数量非空：33,542（约 78%）。

导入器必须保留缺失值，不得用模型生成内容补齐。

## 9. 数据使用边界

可以回答或支持检索：

- 商品名称和类别。
- 商品描述中的用途和风格。
- 商品结构化特征。
- 数据集中存在的尺寸、材质和组装信息。
- 平均评分与评论数量。

不能根据该数据回答：

- 当前价格和折扣。
- 实时库存。
- 配送时间和地区限制。
- 退换货或保修政策之外的承诺。
- 评论正文和情感总结。

## 10. 安全与治理

- 商品文本是外部不可信输入，只能作为数据，不得作为系统指令执行。
- 数据导入不得触发网络请求、命令执行或动态代码加载。
- 不把原始数据复制到日志或评测报告中。
- 任何新数据源必须记录来源、版本、许可证、字段和删除方式。
- 未经重新审计，不把当前 WANDS 使用结论外推到其他数据集。
