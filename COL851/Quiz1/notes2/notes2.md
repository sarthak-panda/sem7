DNNs can be expressed as a directed acyclic compu-
tational graph (DAG), in which nodes represent the opera-
tors (e.g., convolution, matrix multiplication) and directed
edges represent the dependencies between operators.
. 
Existing deep learning frameworks (e.g., Tensorflow [1], PyTorch [39],
MXNet [10]) map the operators in DNNs to vendor-provided
kernel libraries (e.g., cuDNN [13], MKL-DNN [27]) to
achieve high performance. However, these kernel libraries
require significant engineering effort to manually tune for
each hardware platform and operator. The significant manual
effort required to produce efficient operator implementations
for each target accelerator limits the development and innova-
tion of new operators [7] and specialized accelerators [35].

Given the importance of DNNs' performance, researchers
and industry practitioners have turned to search-based com-
pilation [2, 11, 32, 49, 59] for automated generation of tensor
programs, i.e., low-level implementations of tensor operators.
For an operator or a (sub-)graph of multiple operators, users
define the computation in a high-level declarative language
(§2), and the compiler then searches for programs tailored
towards different hardware platforms.

To find performant tensor programs, it is necessary for a
search-based approach to explore a large enough search space
to cover all the useful tensor program optimizations. However,
existing approaches fail to capture many effective optimiza-
tion combinations, because they rely on either predefined
manually-written templates (e.g., TVM [12], FlexTensor [59])
or aggressive pruning by evaluating incomplete programs
(e.g., Halide auto-scheduler [2]), which prevents them from
covering a comprehensive search space (§2). The rules they
use to construct the search space are also limited

In this paper, we explore a novel search strategy for gener-
ating high-performance tensor programs. It can automatically
generate a large search space with comprehensive coverage of
optimizations and gives every tensor program in the space a
chance to be chosen. It thus enables to find high-performance
programs that existing approaches miss

Realizing this goal faces multiple challenges. First, it re-
quires automatically constructing a large search space to cover
as many tensor programs as possible for a given computation
definition. Second, we need to search efficiently without com-
paring incomplete programs in the large search space that can
be orders of magnitude larger than what existing templates
can cover. Finally, when optimizing an entire DNN with many
subgraphs, we should recognize and prioritize the subgraphs
that are critical to the end-to-end performance.

To this end, we design and implement Ansor, a framework
for automated tensor program generation. Ansor utilizes a
hierarchical representation to cover a large search space. This
representation decouples high-level structures and low-level
details, enabling flexible enumeration of high-level structures
and efficient sampling of low-level details. The space is con-
structed automatically for a given computation definition.
Ansor then samples complete programs from the search space
and fine-tunes these programs with evolutionary search and a
learned cost model. To optimize the performance of DNNs
with multiple subgraphs, Ansor dynamically prioritizes sub-
graphs of the DNNs that are more likely to improve the end-
to-end performance.

 Ansor searches more efficiently, generating
higher-performance programs in a shorter time, despite its
larger search space. Ansor can match the performance of a
state-of-the-art framework with an order of magnitude less
search time. Besides, Ansor enables automatic extension to
new operators by only requiring their mathematical definitions
without manual templates.

---------------------------------------\
In summary, this paper makes the following contributions:
• A mechanism to generate a large hierarchical search
space of tensor programs for a computational graph.
• An evolutionary strategy with a learned cost model to
fine-tune the performance of tensor programs.
• A scheduling algorithm based on gradient descent to
prioritize important subgraphs when optimizing the end-
to-end performance of DNNs.
• An implementation and comprehensive evaluation of the
Ansor system demonstrating that the above techniques
outperform state-of-the-art systems on a variety of DNNs
and hardware platforms
----------------------------------------\


2. Background
Template-guided search. In template-guided search, the
search space is defined by manual templates. As shown in Fig-
ure 2a, the compiler (e.g., TVM) requires the user to manually
write a template for a computation definition. The template
defines the structure of the tensor programs with some tunable
parameters (e.g., tile size and unrolling factor). The compiler
then searches for the best values of these parameters for a spe-
cific input shape configuration and a specific hardware target.
This approach has achieved good performance on common
deep learning operators. However, developing templates re-
quires substantial effort. For example, the code repository of
TVM already contains more than 15K lines of code for these
templates. This number continues to grow as new operators
and new hardware platforms emerge. Besides, constructing a
quality template requires expertise in both tensor operators
and hardware. It takes non-trivial research effort [32, 55, 59]
to develop quality templates. Despite the complexity of tem-
plate design, manual templates only cover limited program
structures because manually enumerating all optimization
choices for all operators is prohibitive. This approach typi-
cally requires defining one template for each operator. Flex-
Tensor [59] proposes a general template to cover multiple
operators, but its template is still designed for single operator
granularity, which fails to include optimizations involving
multiple operators (e.g., operator fusion). The search space
of optimizing a computational graph with multiple operators
should contain different ways to compose the operators. A
template-based approach fails to achieve this because it can-
not break down their fixed templates and re-compose them
