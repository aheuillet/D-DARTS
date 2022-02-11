from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

Genotype_nested = namedtuple('Genotype_nested', 'genes concat reductions')

Genotype_opt = namedtuple('Genotype_opt', 'genes seq concat reductions')

"""
Operation sets
"""
PRIMITIVES = [
    'conv_3x1_1x3',
    'conv_7x1_1x7',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'simple_conv_1x1',
    'simple_conv_3x3',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'bottleneck_1x3x1',
]

PRIMITIVES_DARTS = [
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]

"""====== Different Baseline Archirtectures"""

residual_layer = [('bottleneck_1x3x1', 2, 0), ('skip_connect', 2, 0), ('bottleneck_1x3x1', 3, 2), ('skip_connect', 3, 2), ('bottleneck_1x3x1', 4, 3), ('skip_connect', 4, 3)]

residual_layer_simple = [('simple_conv_3x3', 2, 0), ('simple_conv_3x3', 3, 2), ('skip_connect', 3, 0), ('simple_conv_3x3', 4, 3), ('simple_conv_3x3', 5, 4), ('skip_connect', 5, 3)]

ResNet18 = Genotype_opt([residual_layer_simple], [0]*4, concat=range(2, 6), reductions=range(1, 4))

ResNet50 = Genotype_opt(
  [
    residual_layer,
    residual_layer + [('bottleneck_1x3x1', 5, 4), ('skip_connect', 5, 4)],
    residual_layer + [('bottleneck_1x3x1', 5, 4), ('skip_connect', 5, 4), ('bottleneck_1x3x1', 6, 5), ('skip_connect', 6, 5), ('bottleneck_1x3x1', 7, 6), ('skip_connect', 7, 6)]
  ], [0, 1, 2, 0], concat=range(2,8), reductions=range(1, 4))

Xception = Genotype_opt( #TODO: Add Conv 1x1 in skip connect edges
  [[
    ('simple_conv_1x1', 4, 0),
    ('sep_conv_3x3', 2, 0),
    ('sep_conv_3x3', 3, 2),
    ('max_pool_3x3', 4, 3)
    ],
    [
    ('skip_connect', 4, 0),
    ('sep_conv_3x3', 2, 0),
    ('sep_conv_3x3', 3, 2),
    ('sep_conv_3x3', 4, 3)
    ],
    [
    ('sep_conv_3x3', 2, 0),
    ('sep_conv_3x3', 3, 2)
    ]], [0]*3+[1]*8+[0]+[2], concat=range(2,6), reductions=[1, 2, 11, 12])

InceptionV4 = Genotype_opt([
  [
    ('simple_conv_1x1', 2, 0),
    ('avg_pool_3x3', 3, 0),
    ('simple_conv_1x1', 4, 0),
    ('simple_conv_1x1', 6, 0),
    ('simple_conv_1x1', 6, 3),
    ('simple_conv_3x3', 5, 4),
    ('simple_conv_3x3', 6, 5),
    ('simple_conv_3x3', 6, 2)
  ],
  [
    ('simple_conv_1x1', 2, 0),
    ('simple_conv_3x3', 3, 2),
    ('simple_conv_3x3', 4, 3),
    ('max_pool_3x3', 4, 0)
  ],
  [
    ('simple_conv_1x1', 2, 0),
    ('simple_conv_1x1', 3, 0),
    ('avg_pool_3x3', 4, 0),
    ('simple_conv_1x1', 6, 0),
    ('conv_7x1_1x7', 6, 3),
    ('conv_7x1_1x7', 5, 2),
    ('conv_7x1_1x7', 6, 5),
    ('simple_conv_1x1', 6, 4),
  ],
  [
    ('simple_conv_1x1', 2, 0),
    ('simple_conv_3x3', 5, 2),
    ('simple_conv_1x1', 3, 0),
    ('conv_7x1_1x7', 4, 3),
    ('simple_conv_3x3', 5, 4)
  ], 
  [
    ('simple_conv_1x1', 2, 0),
    ('simple_conv_1x1', 4, 0),
    ('avg_pool_3x3', 5, 0),
    ('simple_conv_1x1', 6, 0),
    ('conv_3x1_1x3', 3, 2),
    ('conv_3x1_1x3', 6, 3),
    ('conv_3x1_1x3', 6, 4),
    ('simple_conv_1x1', 6, 5)
  ]]
, [0]*4+[1]+[2]*7+[3]+[4]*3, concat=range(2,7), reductions=[4, 12])

MobileNetV3= Genotype_nested([], concat=range(2,6), reductions=[])

EfficientNetB1 = Genotype_nested([], concat=range(2,6), reductions=[])
