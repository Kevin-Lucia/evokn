"""
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
'prune' attributes (descriptions below).
"""

import numpy as np
import concurrent.futures as cf


class Afferent:

    def __init__(self, prototype, label, items, flow=None, epoch=None,
                 evo_n=None, prune=None, k=2, distance=None, t=None, depth=None,
                 rp=True, r_evo=None, nan=100, col_or=True):
        """
        items      : Data entries with fields of prototype. 2D numpy array.
        k          : Maximum number of nearest neighbors, and nearest-nearest.
        distance   : Inverse of association for attributes. [0,1] scale. Exact
                     likeness would be 0. The extent of neighbor-neighbor
                     distance is 1.
        t          : Threshold--quantity of attributes that must fit distance
                     requirement for a neighbor to be activated.
        depth      : Rounds of nearest neighbor selection, that is not restarted
                     after each nearest neighbor activation for non-rapid
                     evolution. For rapid-evolution, depth is initiated for the
                     nearest neighbor at the level of activation (see revo()).
        rp         : Refractory period--non-rapid neighbors are refractory if
                     activated at any depth. Rapidly evolving systems are
                     refractory only during a single frame. The frame of the
                     prototype shifts in time with every level of depth
                     during rapid evolution.
        If 'None', a default will be calculated for 'distance', 't', 'depth'.
        The default is oriented towards non-rapid evolution.

         prototype  : Numpy array of 2D initial parameters for non-rapid
                      evolution, or the entire time series for rapid *nXtraits*.
                      New members from evolution are added to the bottom of the
                      prototype.
         label      : Numpy array of 2D labeled/scored data *nX1*.
         flow       : Number of items evaluated before prototype evolves. If set
                      to None, then all items added are evaluated and then
                      evolve (if evolution parameters are set).
         epoch      : During non-rapid evolution 'epoch' establishes a number to
                      index the self.labeled_items list portion added to the
                      prototype, or which type of items to 'prune'--referencing
                      that same list.
                      Setting to 0 will start at the top--newest--of the
                      labeled items, and any negative starts from the back.
                      However, note that with negative values you are indexing
                      from a constant item (at the end) and there will be no
                      added information by running learn() and evolving()
                      without constantly modifying 'epoch' (which is possible,
                      and on some occasions desirable). That is, indexing from 0
                      always pulls information from newly introduced members to
                      the population.
                      Between 'epoch', 'evo_n', and 'prune' any subsection can
                      be reached.
                      During rapid evolution, epoch is the size of one frame
                      within a prototype (all frames). For example, one frame
                      could be the specific point in time at which multiple
                      stable states exist.
         evo_n      : Number of newly labeled items to add to prototype, and
                      label. Collected from the start of epoch, down. Or in any
                      order calling evolving() multiple times. Items are added
                      to the bottom of the prototype.
         prune      : Number of items removed from the current prototype as
                      they first appear.
                      Removed items follow a multiple of the newly classified
                      labels starting from the current epoch. Prune is for
                      non-rapid evolution.

                      'flow', 'prune', 'epoch', and 'evo_n' are adjustable
                      at any time.

         r_evo      : For rapidly evolving tasks using learn() to enter revo(),
                      the method for rapid evolution. It is also a multiplier
                      for direct associations of the candidate with prototype.
                      Entering 1 would multiply the score by 1 (accepts any
                      number).
         nan        : Unclassified. The maximum allowable items unable to be
                      labeled, before allowing evolution or pruning for
                      non-rapid tasks. Rapid tasks anticipate large quantities
                      of misses.
         col_or     : Column order: If set to True the same column order used
                      for multiprocessing in learn() is used. See the
                      self.items_labeled placeholder below. False for not using
                      learn() or not using a similar order column.
         Default for these items is left as is.
         """

        self.prototype = prototype
        self.label = label
        """
        For non-rapid evolution, it is a non-permanent self.items list.
        """
        self.items = items

        self.flow = flow
        self.epoch = epoch
        self.evo_n = evo_n
        self.prune = prune

        self.k = k

        if not distance and distance != 0:
            self.distance = 1 / np.unique(self.label).shape[0]
        else:
            self.distance = distance

        if not t and t != 0:
            self.threshold = self.prototype.shape[1] / \
                                      np.unique(self.label).shape[0]
        else:
            self.threshold = t

        """Depth/levels of search. The neurons that the item activates directly
        is depth 1. The neurons the activated, activate (if not refractory) is 
        at depth 2."""
        if not depth and depth != 0:
            self.depth = np.unique(self.label).shape[0]
        else:
            self.depth = depth

        self.rp = rp

        self.r_evo = r_evo

        self.nan = nan

        self.col_or = col_or

        """
        Placeholder for item, order, label/score--in that order
        In rapidly-evolving systems, the last column becomes a score
        In non-rapid systems, last column holds any label.
        ** Needs removal if not using learn() **.
        """
        if self.col_or:
            self.items_labeled = np.zeros((1, prototype.shape[1] + 2))
        else:
            self.items_labeled = np.zeros((1, prototype.shape[1] + 1))

        "Placeholder for result(), labels/scores, ordered by evaluation"
        self.item_labels = None

        "Newly ingested references/data count running through learn()"
        self.new_items = 0

        """Placeholder when restarting learn() with non-rapid tasks, in the 
        middle of a period set to contribute to evolution."""
        self.flow_pool = 0


    def evo(self, sensation):
        """For non-rapid evolution,
        it can be useful for a system to change perspective based on the natural
        drift of attributes. The attributes of any subject evolve over time.
        Most likely it acquires more prototypical attributes, but is certain to
        have a different value for previous ones.
        evo() can be used with evolving() for a prototype which ingests its
        newly labeled data. The method learn() already incorporates the two, if
        evolving.
        Set col_or to None if not using an order column. -1, if last column."""

        if self.col_or:
            col_or = -1
        else:
            col_or = None

        "Normalizing the prototype"
        edge_one = self.prototype.min(axis=0)
        edge_two = self.prototype.max(axis=0)

        "Working memory for a parallel searches under refractory constraints"
        work_memory_P = (self.prototype - edge_one) / (edge_two - edge_one)
        work_memory_L = self.label

        "Normalization for unknown sample"
        stimulus = (sensation[:, :col_or] - edge_one) / (edge_two - edge_one)

        "Initialize labels (neighbor labels) and depth for the stimulus"
        axon_labels = np.array([])
        axons = np.array([])
        current_depth = 0

        while current_depth < self.depth:

            if axons.size:
                stimuli = axons
                axons = np.array([])
                current_depth += 1

            elif current_depth >= 1:
                break

            elif current_depth == 0:
                stimuli = stimulus
                current_depth += 1

            for synapse in stimuli[:, None]:

                proximity = np.absolute(work_memory_P - synapse)
                dendrite = np.where(proximity <= self.distance, 1, 0)
                hillock = np.sum(dendrite, axis=1, keepdims=True)

                "Nearest at start and closer together, for further depths"
                if current_depth == 1 and self.depth > 1:
                    hillock_wm = np.hstack((work_memory_P,
                                            work_memory_L,
                                            hillock))
                    ordered_h_wm = hillock_wm[np.argsort(-1*hillock_wm[:, -1])]
                    work_memory_P = ordered_h_wm[:, :-2]
                    work_memory_L = ordered_h_wm[:, -2:-1]
                    hillock = ordered_h_wm[:, -1]

                "Loop for labeling item associations"
                for neighbor in range(self.k):
                    nearest = np.argmax(hillock)

                    if hillock[nearest] >= self.threshold:
                        activated = work_memory_L[nearest]

                        axon_labels = np.hstack((axon_labels, activated))

                        "An axons list for next depths to iterate"
                        if current_depth != self.depth:
                            potential = work_memory_P[[nearest]]

                            if axons.size:
                                axons = np.vstack((axons, potential))
                            else:
                                axons = potential

                            "Refractory period"
                            if self.rp:
                                work_memory_P = np.delete(work_memory_P,
                                                          obj=nearest,
                                                          axis=0)
                                work_memory_L = np.delete(work_memory_L,
                                                          obj=nearest,
                                                          axis=0)
                                hillock = np.delete(hillock,
                                                    obj=nearest,
                                                    axis=0)
                            else:
                                np.put(hillock, nearest, 0)
                    else:
                        break

        # In line to follow while loop
        "Commanding stimulus attributed; highest type of triggered axons"
        if axon_labels.size:
            # 1D activated, count
            activated, count = np.unique(axon_labels, return_counts=True)

            "Item label. 2D to fit on self.items_labeled and prototype"
            item_label = activated[[[np.argmax(count)]]]
            item_labeled = np.hstack((sensation, item_label))

        else:
            item_labeled = np.hstack((sensation, [[np.nan]]))

        "For parallel processes not using learn()"
        self.items_labeled = np.vstack((item_labeled,
                                        self.items_labeled))
        return item_labeled


    def evolving(self, epoch=None, evo_n=None, prune=None):
        """
        Evolve prototype,
        according to the position and occurrence of the labeled items;
        evolving() is a method to shift a prototype's changing attributes, to
        accommodate natural changes with time. It is used in learn() for
        non-rapid evolution.
        Although evolving() works with learn(), it can be used by itself.
        Methods at the end of the program have been added to rapidly adjust the
        parameters here (or in learn()).

        Setting epoch to 0, copies items to the prototype and label starting
        from the newest item labeled; or (if pruning) removes from the
        prototype, the first occurrence of the label given to the newest item.

        Any portion of new items can be referenced. Setting epoch to negative
        values starts from the last of items labeled, and works down.
        * Note: starting from the back is always starting from the oldest item
        here *

        Neither adding nor removing items from the prototype while evolving()
        precludes the other.
        """

        if self.col_or:
            col_or = -2
        else:
            col_or = -1

        if epoch or epoch == 0:
            self.epoch = epoch
        if evo_n:
            self.evo_n = evo_n
        if prune:
            self.prune = prune

        """Items unable to be labeled affect the order; Total allowed is saved 
        in self.nan"""
        clean_list = \
            self.items_labeled[~np.isnan(self.items_labeled).any(axis=1)]

        unclassified = self.items_labeled.shape[0] - clean_list.shape[0]

        assert unclassified <= self.nan

        "Prune first to avoid eliminating new prototype members"
        if self.prune:
            pruned = 0
            pruning = self.epoch
            while pruned < self.prune:
                """Index item's label from start of epoch where it first appears
                in prototype's label."""
                try:
                    if ~np.isnan(self.items_labeled[pruning]).any():
                        index = np.where(self.label ==
                                         self.items_labeled[pruning, -1])[0][0]

                        self.label = np.delete(self.label, obj=index,
                                               axis=0)
                        self.prototype = np.delete(self.prototype, obj=index,
                                                   axis=0)
                        pruned += 1

                except IndexError as ie:
                    if pruning >= self.epoch + 1.1 * self.prune:
                        print("""Unable to fully prune within 110% of the 
                        desired prune range. Check items labeled for nan values.
                        Or there may be not enough new items labeled to 
                        reference removable material""")
                        raise ie
                    else:
                        pass

                pruning += 1

        "Attach new items to prototype amd label"
        if self.evo_n:
            if self.epoch + self.evo_n == 0:
                self.prototype = np.vstack((self.prototype,
                                            clean_list[self.epoch:, :col_or]))
                self.label = np.vstack((self.label,
                                        clean_list[self.epoch:, -1:]))
            else:
                self.prototype = np.vstack((self.prototype,
                     clean_list[self.epoch:(self.epoch + self.evo_n), :col_or]))

                self.label = np.vstack((self.label,
                        clean_list[self.epoch:(self.epoch + self.evo_n), -1:]))


    def revo(self, candidate):
        """
        When prototypes evolve rapidly,
        how the neighbors compare to future elements takes on more meaning. If
        'A' is close to 'B now', in a snapshot of the expected behavior, then if
        'B (past)' is closer to high value 'C' in a future expected model--'A'
        deserves a higher score. This model was designed to account for the
        value in tracking all stable states through time. Parts of the frame
        that are compared to the item being evaluated, are in turn compared to
        future parts of the frame. If they are close to themselves in the
        future: valuable. If they are close to higher valued items in the
        future: valuable. If the last frame is similar to the
        first: valuable. All of these scenarios follow because a split second
        earlier, or later, the item under comparison was similar.
        Set 'epoch' to the frame length. The entire prototype should be evenly
        divisible by 'epoch'.
        """
        if self.col_or:
            col_or = -1
        else:
            col_or = None

        "Normalize according to first frame"
        edge_one = self.prototype[:self.epoch].min(axis=0)

        edge_two = self.prototype[:self.epoch].max(axis=0)

        work_memory_P = (self.prototype[:self.epoch] - edge_one) / \
                        (edge_two - edge_one)

        work_memory_L = self.label[:self.epoch]

        stimulus = (candidate[:, :col_or] - edge_one) / (edge_two - edge_one)

        "Initialize score for the candidate"
        value = 0

        "After last frame, shifts back to beginning to score last neighbors"
        end = False

        "Placeholder for activated neighbors, depth information in last column"
        axons = np.array([])
        axons_d = np.array([])

        "Checks all the frames"
        frames = self.prototype.shape[0] / self.epoch
        frame = 0

        """
        Term for neighbors in last frames to loop back on first. No new
        neighbors are added when returning to original frames, but count 
        continues from the neighbors only, until they finish full depth.
        """
        loop_around = None

        while loop_around != 0:
            if end:
                loop_around -= 1
            """
            axons_d (last column is depth) is cleaned at the bottom of loop. 
            The depth at which neighbors (axons) are added is their first depth.
            """
            if not axons_d.size:
                stimuli = stimulus
                cleft = 0
            else:
                if not end:
                    stimuli = np.vstack((stimulus, axons_d[:, :-1]))
                    cleft = 0
                else:
                    stimuli = axons_d[:, :-1]

            for synapse in stimuli[:, None]:
                cleft += 1

                proximity = np.absolute(work_memory_P - synapse)
                dendrite = np.where(proximity <= self.distance, 1, 0)
                hillock = np.sum(dendrite, axis=1)

                "Loop for adding value and adding neighbors "
                for neighbor in range(self.k):

                    try:
                        nearest = np.argmax(hillock)

                    except ValueError as ve:
                        print(f"""
                        ValueError raised: 
                        Activation may have amplified to make the whole frame 
                        refractory. Raise the 'threshold', decrease 'distance' 
                        lower 'k' (nearest neighbors), or increase frame size.
                        Initiated by candidate {candidate} 
                        ** item number {candidate[:, -1]} **""")
                        raise ve


                    if hillock[nearest] >= self.threshold:
                        "Direct hits can be weighted"
                        if cleft == 1:
                            value += work_memory_L[nearest] * self.r_evo
                        else:
                            value += work_memory_L[nearest]

                        if not end:
                            "An axons list for next depths to iterate"
                            potential = work_memory_P[[nearest]]

                            if not axons.size:
                                axons = potential
                            else:
                                axons = np.vstack((axons, potential))

                        """Refractory period only applies for nearest neighbor
                        searches at the current frame, which changes each level
                        of depth (for rapid-evolution)"""
                        if self.rp:
                            work_memory_P = np.delete(work_memory_P,
                                                      obj=nearest, axis=0)

                            work_memory_L = np.delete(work_memory_L,
                                                      obj=nearest, axis=0)

                            hillock = np.delete(hillock, obj=nearest, axis=0)
                        else:
                            np.put(hillock, nearest, 0)

                    else:
                        break
            # In line with outer for loop
            if axons.size:

                if not end:
                    d = np.zeros((axons.shape[0], 1))
                    new_axons = np.hstack((axons, d))
                    axons = np.array([])

                    if axons_d.size:
                        axons_d = np.vstack((axons_d, new_axons))
                    else:
                        axons_d = new_axons
            """
            The depth at activation is always considered the first depth for 
            that neighbor.
            """
            if axons_d.size:
                axons_d[:, -1] = axons_d[:, -1] + 1
                axons_d = axons_d[axons_d[:, -1] < self.depth]

            "Frame count used in 'Frame shift'"
            frame += 1

            """Section to initiate return to 1st frame for neighbors still 
            activated in the last frame. 
            No new neighbors are added after looping back to first frame, but 
            nearest-nearest associations are scored. The candidate is not."""
            if frame == frames:
                if not axons_d.size:
                    break
                elif not end:
                    loop_around = self.depth - axons_d[-1, -1]

                end = True
                frame = 0

            "Frame shift"
            "Invert before re-normalizing with respect to new frame"
            if axons_d.size:
                axons_d[:, :-1] = (axons_d[:, :-1] * (edge_two - edge_one)) + \
                                                                       edge_one
            edge_one = self.prototype[frame * self.epoch:
                                      self.epoch * (1 + frame)].min(axis=0)

            edge_two = self.prototype[frame * self.epoch:
                                      self.epoch * (1 + frame)].max(axis=0)
            if axons_d.size:
                axons_d[:, :-1] = (axons_d[:, :-1] - edge_one) / \
                                    (edge_two - edge_one)

            work_memory_P = (self.prototype[frame * self.epoch:
                             self.epoch * (1 + frame)] - edge_one) / \
                            (edge_two - edge_one)

            work_memory_L = self.label[frame * self.epoch:
                                       self.epoch * (1 + frame)]

            stimulus = (candidate[:, :col_or] - edge_one) / \
                       (edge_two - edge_one)

        # In line to follow the While loop.
        if value:
            item_labeled = np.hstack((candidate, [value]))
        else:
            item_labeled = np.hstack((candidate, [[np.nan]]))

        "In case not using learn()"
        self.items_labeled = np.vstack((item_labeled,
                                        self.items_labeled))

        return item_labeled


    def add_new(self, new):
        """Standalone method to add new items and continue object use. 2D numpy
        array of attributes only"""

        if self.items.size:
            self.items = np.vstack((self.items, new))
        else:
            self.items = new


    def learn(self, flow=False, nodes=False, pool_warning=True, r_evo=None,
              epoch=None, evo_n=None, prune=None):
        """
        This method is designed to handle multiprocessing of rapidly and
        non-rapidly evolving systems. Evolution parameters may be off and still
        run this method. To use rapid-evolution (revo()), set 'r_evo'.

        'r_evo'         : a multiplier for direct hits (1 if same score as
                          neighbor hits).

        'nodes'         : the maximum number of processors (1-61).'None' sets to
                          number of cores on the computer.

        'pool_warning'  : for a printed reminder of items remaining  after
                          labeling finishes, but flow to evolution has not.

        This method washes the self.items list, after items are processed but
        not before 'flow' has been filled--for non-rapid evolution. This was
        done in an attempt to give objects more persistence;rather than for
        single use evaluation, the prototype seems more valuable always
        receiving new items, and evolving over time.
        The 'self.items_labeled' list holds all processed items. To add new
        items to 'self.items' use the above method, add_new().
        """
        if flow or flow is None:
            self.flow = flow
        if r_evo:
            self.r_evo = r_evo
        if epoch or epoch == 0:
            self.epoch = epoch
        if evo_n:
            self.evo_n = evo_n
        if prune:
            self.prune = prune

        if nodes:
            nodes = round(nodes)
            if nodes > 61:
                nodes = None

        warning_count = 0

        evaluating = self.items

        while evaluating.size:

            if self.r_evo or not self.flow:
                flowing = evaluating
                evaluating = np.array([])
            else:
                flowing = evaluating[self.flow_pool:self.flow, :]
                self.flow_pool += flowing.shape[0]
                evaluating = evaluating[self.flow:, :]

            "Creating the preserve order column"
            first_item = self.new_items
            self.new_items += flowing.shape[0]
            col_or = np.arange(first_item + 1, self.new_items + 1, 1)[:,
                                                                     np.newaxis]
            flow_n = np.hstack((flowing, col_or))

            "Distributing nodes, consider 'max_workers' maximum in docs"
            if not nodes and nodes is not None:
                nodes = flow_n.shape[0]
                if nodes > 61:
                    nodes = None

            "revo() or evo()"
            if self.r_evo:
                with cf.ProcessPoolExecutor(max_workers=nodes) as executor:
                    futures = [executor.submit(self.revo, item_n)
                               for item_n in flow_n[:, None]]

                    for labeled in cf.as_completed(futures):
                        self.items_labeled = np.vstack((labeled.result(),
                                                        self.items_labeled))

            else:
                with cf.ProcessPoolExecutor(max_workers=nodes) as executor:
                    futures = [executor.submit(self.evo, item_n)
                               for item_n in flow_n[:, None]]

                    for labeled in cf.as_completed(futures):
                        self.items_labeled = np.vstack((labeled.result(),
                                                        self.items_labeled))

            if first_item == 0:
                self.items_labeled = self.items_labeled[:-1, :]

            self.items_labeled = self.items_labeled[
                                    np.argsort(-1 * self.items_labeled[:, -2])]

            if not self.r_evo:

                if not self.flow:
                    self.items = np.array([])
                    if self.evo_n or self.prune:
                        self.evolving()

                elif self.flow_pool == self.flow:
                    self.items = self.items[self.flow:]
                    self.evolving()
                    self.flow_pool = 0

        # In line to follow while loop
        if self.flow:
            if self.flow_pool != self.flow and self.flow_pool != 0:
                if pool_warning:
                    warning_count += 1
                    print(f"""
            {self.flow_pool} item(s) remain in the pool for evolution; labeled 
            but not used in the current 'flow' for evolving. Pooled items will 
            be used for evolving after the flow (currently set to {self.flow}) 
            has been filled (see self.items for the pool).
            
            **Use add_new() to add more items** 
            
            If items are added to self.items directly, the current remainder 
            will be lost for evolution. In that case, the self.flow_pool will 
            need to be reset to 0, or {self.flow_pool} will be lost from 
            that directly added list for labeling as well as evolution.
            Warning count {warning_count}
            """)


    def result(self, ord_list=False, raw_list=False):
        """Returns the labels or scores in the order provided"""

        if self.col_or:
            col_or = -2
        else:
            col_or = -1

        early_order = self.items_labeled[
                                     np.argsort(self.items_labeled[:, -2])]

        if ord_list:
            ord_list = np.delete(early_order, obj=col_or, axis=1)
            result = ord_list
        elif raw_list:
            result = self.items_labeled
        else:
            self.item_labels = early_order[:, -1:]
            result = self.item_labels
        print(result)
        return result


    def percent_true(self, true):
        """Takes as an argument 'true', 2D numpy array--nx1--of labels only"""

        early_order = self.items_labeled[
                                  np.argsort(self.items_labeled[:, -2])]

        predicted = early_order[:, -1:]
        correct = np.where(true == predicted, 1, 0)
        score = np.sum(correct)
        percent = score / true.shape[0]
        print(percent*100)
        return percent*100


    def evolve(self, flow=False, epoch=None, evo_n=None, prune=None):
        """Method to toggle evolution"""

        if epoch:
            self.epoch = epoch
        if evo_n:
            self.evo_n = evo_n
        if flow or flow is None:
            self.flow = flow
        if prune:
            self.prune = prune


    def prune(self, flow=False, prune=None):
        """Method to toggle pruning"""

        if prune:
            self.prune = prune
        if flow or flow is None:
            self.flow = flow


    def evo_p(self, flow=False, k=None, distance=None, t=None, depth=None,
              rp=None, r_evo=None, nan=None, col_or=None):
        """Method for hyperparameter tuning"""

        if flow or flow is None:
            self.flow = flow
        if k:
            self.k = k
        if distance:
            self.distance = distance
        if t:
            self.threshold = t
        if depth:
            self.depth = depth
        if rp:
            self.rp = rp
        if r_evo:
            self.r_evo = r_evo
        if nan:
            self.nan = nan
        if col_or:
            self.col_or = col_or
