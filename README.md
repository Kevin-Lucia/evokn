# evokn
Evolving k-nearest neighbors

In this implementation,

of k-nearest (kn) neighbors an item is assessed for its deviation from evolving
prototype attributes. If close enough to an attribute, then a dendrite for the
prototype is activated. Enough dendrites/attributes activated, will allow the
nearest neighbor holding those attributes to reach threshold.
Another kn search is initiated from the neurons/neighbors that reach threshold.
At every level of 'depth', class labels are summed from each of the neurons that
reach 'threshold'. A 'refractory period' can be utilized to prevent recurrent
activations during each item's association, for every level of depth during
non-rapid evolution. During rapid evolution, the entire prototype changes--frame
by frame--with each level of depth; and is only refractory during the current
frame for activated neighbors.


Matching narrow requirements,

is assumed to be the purpose of evaluating rapidly evolving tasks. For example,
matching a fit for a landmark in an enzyme target pocket with drug candidates.
If the following were utilized for such a purpose, then an object would be made
for each landmark's ideal traits. A score based system is adopted for rapidly
evolving tasks (see revo()).
Though both rapid and non-rapid prototypes are set to evolve, rapid prototypes
immediately change the entire prototype--which are frames in time (any amount of
time). All the evolution for rapid tasks is written into revo().


Adapting for a changing environment,

is the aim of the non-rapid evolutionary kn search implemented. The items that
are classified eventually become the prototype for comparing things to follow.
It seems to be the natural order of things to change in value over time. The
rate of prototype changes can be adjusted with 'flow', 'epoch', 'evo_n', and
'prune' attributes.
