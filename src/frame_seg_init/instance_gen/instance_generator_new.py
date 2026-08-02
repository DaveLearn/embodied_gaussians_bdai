from typing import Dict, List, Optional, Tuple
import torch

from frame_seg_init.instance_gen.edge_matcher import (
    EdgeScoreMethod,
    global_best_matching,
)
from frame_seg_init.instance_gen.frame_seg_scene_graph import FrameSegSceneGraph

import logging

logger = logging.getLogger(__name__)


class EdgeCompatibility:
    SEGMENT_A = 0
    SEGMENT_B = 1
    FRAME_A_IDX = 2
    FRAME_B_IDX = 3

    def __init__(self, device: str | torch.device = "cuda"):
        self.device = device
        self.compatible: torch.Tensor = torch.empty(
            (0, 4), device=device, dtype=torch.int
        )  # (num_edges, (segment_id_a, segment_id_b, frame_a, frame_b))
        self.all_scores: torch.Tensor = torch.empty(
            (0, 6), device=device, dtype=torch.float
        )  # (num_edges, (segment_id_a, segment_id_b, frame_a, frame_b, score, intersect_count))
        self.children: torch.Tensor = torch.empty((0, 2), device=device, dtype=torch.int)  # (num_edges, (segment_id_parent, segment_id_child))

    def build_from_sg(
        self,
        sg: FrameSegSceneGraph,
        intersection_thresh: int,
        edge_score_method: EdgeScoreMethod,
    ):
        frame_ids = [frame.id for frame in sg.frames]

        compatible_chunks: List[torch.Tensor] = []
        all_score_chunks: List[torch.Tensor] = []
        children_chunks: List[torch.Tensor] = []

        for idx_a, frame_A_id in enumerate(frame_ids):
            frame_A_segment_ids = sg.frame_segs.ids[sg.frame_segs.frame_ids == frame_A_id]

            # build children which are just edges which have intersection_frac > 0.95 in the same frame

            child_mask = (
                (sg.edges.intersection_frac > 0.85)
                & (torch.isin(sg.edges.frame_seg_ids[:, 0], frame_A_segment_ids))
                & (torch.isin(sg.edges.frame_seg_ids[:, 1], frame_A_segment_ids))
            )

            child_pairs = sg.edges.frame_seg_ids[child_mask]
            if child_pairs.shape[0] > 0:
                # Ensure parent (col 0) has area >= child (col 1); swap if not
                areas_0 = sg.frame_segs.get_areas_by_ids(child_pairs[:, 0])
                areas_1 = sg.frame_segs.get_areas_by_ids(child_pairs[:, 1])
                swap = areas_0 < areas_1
                child_pairs[swap] = child_pairs[swap].flip(1)
                # Deduplicate so bidirectional pairs collapse into one
                child_pairs = child_pairs.unique(dim=0)
            if child_pairs.shape[0] > 0:
                children_chunks.append(child_pairs)

            for idx_b, frame_B_id in enumerate(frame_ids):
                # we have already done this pair in the other direction
                if idx_b <= idx_a:
                    continue

                frame_B_segment_ids = sg.frame_segs.ids[sg.frame_segs.frame_ids == frame_B_id]

                matching_pairs = global_best_matching(
                    sg.edges,
                    frame_A_segment_ids,
                    frame_B_segment_ids,
                    min_overlap_count=intersection_thresh,
                    edge_score_method=edge_score_method,
                )

                matching_rows = []
                for pair in matching_pairs:
                    segment_a, segment_b, _score = pair
                    matching_rows.append([segment_a, segment_b, idx_a, idx_b])
                if matching_rows:
                    compatible_chunks.append(torch.tensor(matching_rows, device=self.device, dtype=torch.int))

                # build all scores
                edges = sg.edges
                edge_ids = edges.frame_seg_ids
                mask_ab = torch.isin(edge_ids[:, 0], frame_A_segment_ids) & torch.isin(edge_ids[:, 1], frame_B_segment_ids)
                mask_ba = torch.isin(edge_ids[:, 0], frame_B_segment_ids) & torch.isin(edge_ids[:, 1], frame_A_segment_ids)

                if mask_ab.any() and mask_ba.any():
                    edges_ab = edge_ids[mask_ab]
                    edges_ba = edge_ids[mask_ba]

                    if edges_ab.numel() > 0 and edges_ba.numel() > 0:
                        max_id = torch.max(torch.cat([edges_ab.reshape(-1), edges_ba.reshape(-1)]))
                        key_scale = max_id.to(torch.int64) + 1
                        keys_ab = edges_ab[:, 0].to(torch.int64) * key_scale + edges_ab[:, 1].to(torch.int64)
                        keys_ba = edges_ba[:, 1].to(torch.int64) * key_scale + edges_ba[:, 0].to(torch.int64)

                        keys_ba_sorted, idx_ba_sorted = torch.sort(keys_ba)
                        pos = torch.searchsorted(keys_ba_sorted, keys_ab)
                        valid = (pos < keys_ba_sorted.shape[0]) & (keys_ba_sorted[pos] == keys_ab)

                        if valid.any():
                            pos = pos[valid]
                            edges_ab = edges_ab[valid]
                            idx_ba = idx_ba_sorted[pos]

                            overlap_a = edges.intersection_frac[mask_ab][valid]
                            overlap_b = edges.intersection_frac[mask_ba][idx_ba]
                            intersect_count = torch.minimum(
                                edges.intersection_count[mask_ab][valid],
                                edges.intersection_count[mask_ba][idx_ba],
                            ).to(torch.float)

                            if edge_score_method == "binary":
                                score = torch.ones_like(overlap_a)
                            else:
                                score = overlap_a / (1 + (overlap_a / overlap_b) - overlap_a)

                            seg_a = edges_ab[:, 0].to(torch.float)
                            seg_b = edges_ab[:, 1].to(torch.float)
                            frame_a = torch.full_like(seg_a, float(idx_a))
                            frame_b = torch.full_like(seg_a, float(idx_b))
                            all_score_chunks.append(
                                torch.stack(
                                    [
                                        seg_a,
                                        seg_b,
                                        frame_a,
                                        frame_b,
                                        score,
                                        intersect_count,
                                    ],
                                    dim=1,
                                )
                            )

        if compatible_chunks:
            self.compatible = torch.cat(compatible_chunks, dim=0)
        else:
            self.compatible = torch.empty((0, 4), device=self.device, dtype=torch.int)

        if all_score_chunks:
            self.all_scores = torch.cat(all_score_chunks, dim=0)
        else:
            self.all_scores = torch.empty((0, 6), device=self.device, dtype=torch.float)

        if children_chunks:
            self.children = torch.cat(children_chunks, dim=0)
        else:
            self.children = torch.empty((0, 2), device=self.device, dtype=torch.int)

        logger.info("edge building complete:")
        logger.info(f"compatible edges: {self.compatible.shape[0]}")
        logger.info(f"all scores: {self.all_scores.shape[0]}")
        logger.info(f"children: {self.children.shape[0]} : {self.children}")

    # candidates [N, num_frames], output [N]
    # score is the sum of the edge scores for the candidate
    def compute_candidate_scores(self, candidates: torch.Tensor) -> torch.Tensor:
        scores = torch.zeros((candidates.shape[0]), device=self.device, dtype=torch.float)
        for i in range(self.all_scores.shape[0]):
            seg_a, seg_b, frame_a, frame_b, score, intersect_count = self.all_scores[i]
            mask = (candidates[:, int(frame_a)] == int(seg_a)) & (candidates[:, int(frame_b)] == int(seg_b))
            scores[mask] += score * intersect_count

        return scores


class InstanceCandidates:
    def __init__(self, frame_count: int, device: str | torch.device = "cuda"):
        self.device = device
        self.candidates: torch.Tensor = torch.empty((0, frame_count), device=device, dtype=torch.int)  # (num_candidates, frame_count)

    def add_candidate(self, candidate: torch.Tensor):
        if self.is_subset_of_candidate(candidate):
            return

        self.candidates = torch.cat([self.candidates, candidate.unsqueeze(0)], dim=0)

    def is_subset_of_candidate(self, query_candidate: torch.Tensor) -> bool:
        non_zero_mask = query_candidate != 0
        return bool((self.candidates[:, non_zero_mask] == query_candidate[non_zero_mask]).all(dim=1).any())

    def select_unused_edge(self, edges: EdgeCompatibility) -> Optional[Tuple[int, int, int, int]]:  # (segment_a, segment_b, frame_a, frame_b)
        """
        select an edge that we haven't used yet
        """
        # Direct column indexing since segments map uniquely to frames
        all_values_at_a = self.candidates[:, edges.compatible[:, EdgeCompatibility.FRAME_A_IDX].long()].T  # [M, N]
        all_values_at_b = self.candidates[:, edges.compatible[:, EdgeCompatibility.FRAME_B_IDX].long()].T  # [M, N]

        matches_a = all_values_at_a == edges.compatible[:, EdgeCompatibility.SEGMENT_A].unsqueeze(1)
        matches_b = all_values_at_b == edges.compatible[:, EdgeCompatibility.SEGMENT_B].unsqueeze(1)

        unused_mask = ~(matches_a & matches_b).any(dim=1)

        unused_edges = edges.compatible[unused_mask]

        if unused_edges.shape[0] == 0:
            return None

        return tuple(unused_edges[0].tolist())

    def select_compatible_edge(self, candidate: torch.Tensor, edges: EdgeCompatibility) -> Optional[Tuple[int, int, int, int]]:
        # A compatible edge has one candidate segment and an empty opposite frame.
        compatible_edges = edges.compatible

        # now find the compatible edges that have one segment in the candidate and the other segment is not in the candidate (0)
        a_mask = torch.isin(compatible_edges[:, EdgeCompatibility.SEGMENT_A], candidate) & (
            candidate[compatible_edges[:, EdgeCompatibility.FRAME_B_IDX]] == 0
        )
        b_mask = torch.isin(compatible_edges[:, EdgeCompatibility.SEGMENT_B], candidate) & (
            candidate[compatible_edges[:, EdgeCompatibility.FRAME_A_IDX]] == 0
        )

        compatible_edges_in_candidate = compatible_edges[a_mask | b_mask]

        if compatible_edges_in_candidate.shape[0] == 0:
            return None

        return tuple(compatible_edges_in_candidate[0].tolist())

    def complete_instance(self, candidate: torch.Tensor, edges: EdgeCompatibility) -> torch.Tensor:
        """
        complete an instance by adding edges to the candidate
        """
        while True:
            # find a compatible edge.
            compatible_edge = self.select_compatible_edge(candidate, edges)

            if compatible_edge is None:
                return candidate

            # what segment are we adding?
            if torch.isin(compatible_edge[EdgeCompatibility.SEGMENT_A], candidate):
                new_segment = compatible_edge[EdgeCompatibility.SEGMENT_B]
                new_frame = compatible_edge[EdgeCompatibility.FRAME_B_IDX]
            else:
                new_segment = compatible_edge[EdgeCompatibility.SEGMENT_A]
                new_frame = compatible_edge[EdgeCompatibility.FRAME_A_IDX]

            candidate[new_frame] = new_segment

    def build_from_edges(self, edges: EdgeCompatibility):
        """Build an over-complete set of instance hypotheses from the edge graph.

        Candidates are seeded from compatible edges not yet covered by any
        candidate and grown by attaching edge-connected segments to empty frame
        slots. The loop runs until every compatible edge is covered by at least
        one candidate, so the same physical object usually appears in several
        overlapping candidates; selection later picks one hypothesis per
        segment. This coverage invariant is also what makes consensus trimming
        during selection safe: a segment trimmed from one candidate still rides
        in the candidates that cover its other edges.
        """
        while True:
            unused_edge = self.select_unused_edge(edges)
            if unused_edge is None:
                break

            candidate = torch.zeros((self.candidates.shape[1]), device=self.device, dtype=torch.int)
            candidate[unused_edge[EdgeCompatibility.FRAME_A_IDX]] = unused_edge[EdgeCompatibility.SEGMENT_A]
            candidate[unused_edge[EdgeCompatibility.FRAME_B_IDX]] = unused_edge[EdgeCompatibility.SEGMENT_B]

            self.add_candidate(self.complete_instance(candidate, edges))

        logger.info(f"candidate building complete: {self.candidates.shape[0]} candidates")


class InstanceBuilder:
    def __init__(
        self,
        sg: FrameSegSceneGraph,
        intersection_thresh: int,
        edge_score_method: EdgeScoreMethod,
        element_segments: torch.Tensor,
    ):
        self.sg = sg

        self.intersection_thresh = intersection_thresh
        self.edge_score_method: EdgeScoreMethod = edge_score_method
        self.element_segments = element_segments

        self.instances: InstanceCandidates = InstanceCandidates(len(sg.frames), device=sg.device)
        self.edges = EdgeCompatibility(device=sg.device)

        self.object_instances: torch.Tensor = torch.empty((0, len(sg.frames)), device=self.instances.device, dtype=torch.int)
        self.merged_child_rows: List[Tuple[torch.Tensor, int]] = []  # (child candidate row, position of surviving parent in object_instances)

    def build_edges(self):
        self.edges.build_from_sg(
            self.sg,
            self.intersection_thresh,
            self.edge_score_method,
        )

    def build_instance_candidates(self):
        self.instances.build_from_edges(self.edges)

    # builds self.object_instances from candidates, leaving them in order of best score first
    def select_best_candidates(self):
        """Greedily select the highest-scoring non-conflicting candidates."""
        candidates = self.instances.candidates.clone()
        scores = self.edges.compute_candidate_scores(candidates)

        selected_rows: List[torch.Tensor] = []

        # take top scoring candidate
        while candidates.shape[0] > 0:
            top_score_idx = torch.argmax(scores)
            top_score_candidate = candidates[top_score_idx]

            selected_rows.append(top_score_candidate.clone())
            logger.debug(f"best candidates selected: {top_score_candidate} at score {scores[top_score_idx]}")

            children_mask = torch.isin(candidates, top_score_candidate[top_score_candidate != 0]).any(dim=1)
            candidates = candidates[~children_mask]
            scores = scores[~children_mask]

        if selected_rows:
            self.object_instances = torch.stack(selected_rows)
        else:
            self.object_instances = torch.empty((0, self.instances.candidates.shape[1]), device=self.instances.device, dtype=torch.int)

        logger.info(f"best candidates selected: {self.object_instances.shape[0]} instances")

    def _is_child_segment(self, parent_seg: int, child_seg: int) -> bool:
        children_mask = self.edges.children[:, 0] == parent_seg
        return bool((self.edges.children[children_mask][:, 1] == child_seg).any())

    # best-scoring counterpart of seg in frame_idx according to all_scores
    def _containment_frac(self, child_seg: int, parent_seg: int) -> float:
        # fraction of the child's segment lying inside the parent's, taken from
        # the same-frame edge table built during graph construction; 0 if no edge
        edges = self.sg.edges
        edge_mask = (edges.frame_seg_ids[:, 0] == child_seg) & (edges.frame_seg_ids[:, 1] == parent_seg)
        if not bool(edge_mask.any()):
            return 0.0
        return float(edges.intersection_frac[edge_mask].max())

    def child_instance_score(self, parent_instance: torch.Tensor, child_instance: torch.Tensor) -> float:
        """Vote that child_instance is a duplicate fragment of parent_instance.

        Frames where only the parent appears vote +1, frames where only the
        child appears vote -1, and co-present frames vote +1/-1 on whether the
        child's segment is a registered child of the parent's. The discrete
        vote deadlocks at 0 when an even number of frames vote, so co-present
        frames also contribute a containment-fraction tie-break in (-0.5, 0.5):
        it decides ties but can never overturn a non-zero vote.
        """
        vote = 0
        containment_fracs: List[float] = []
        for frame_idx in range(parent_instance.shape[0]):
            parent_seg = int(parent_instance[frame_idx])
            child_seg = int(child_instance[frame_idx])
            if parent_seg != 0 and child_seg == 0:
                vote += 1
            elif parent_seg == 0 and child_seg != 0:
                vote -= 1
            elif parent_seg != 0 and child_seg != 0:
                vote += 1 if self._is_child_segment(parent_seg, child_seg) else -1
                containment_fracs.append(self._containment_frac(child_seg, parent_seg))

        tie_break = 0.0
        if containment_fracs:
            tie_break = sum(frac - 0.5 for frac in containment_fracs) / len(containment_fracs)

        return vote + tie_break

    def _find_best_parent_for_children(self) -> Dict[int, int]:
        # each child belongs to the parent with the strongest vote; ties keep the
        # earlier (better scored) parent since object_instances is best-first
        best_parent: Dict[int, int] = {}
        for child_idx in range(self.object_instances.shape[0]):
            best_score = 0.0
            for parent_idx in range(self.object_instances.shape[0]):
                if parent_idx == child_idx:
                    continue
                score = self.child_instance_score(self.object_instances[parent_idx], self.object_instances[child_idx])
                if score > best_score:
                    best_score = score
                    best_parent[child_idx] = parent_idx
        return best_parent

    @staticmethod
    def _resolve_surviving_root(child_idx: int, best_parent: Dict[int, int]) -> Optional[int]:
        # follow parent links until reaching an instance that is no one's child,
        # so chained merges (C into B into A) land on A rather than on the
        # also-culled B; a cycle means nothing survives to receive the merge
        idx = child_idx
        visited = {idx}
        while idx in best_parent:
            idx = best_parent[idx]
            if idx in visited:
                return None
            visited.add(idx)
        return idx

    def merge_children(self):
        num_instances = self.object_instances.shape[0]
        best_parent = self._find_best_parent_for_children()

        surviving_idxs = [idx for idx in range(num_instances) if idx not in best_parent]
        surviving_pos = {old_idx: pos for pos, old_idx in enumerate(surviving_idxs)}

        # keep each culled child's segments so assign_instances_to_segments can
        # preserve its coverage in frames where the surviving parent has no segment
        self.merged_child_rows = []
        for child_idx in best_parent:
            root = self._resolve_surviving_root(child_idx, best_parent)
            if root is None:
                continue
            self.merged_child_rows.append((self.object_instances[child_idx].clone(), surviving_pos[root]))

        self.object_instances = self.object_instances[torch.tensor(surviving_idxs, device=self.object_instances.device, dtype=torch.long)]

        logger.info("merging children complete:")
        logger.info(f"child instances merged into parents: {len(self.merged_child_rows)}")
        logger.info(f"child instances culled: {num_instances - len(surviving_idxs)}")
        logger.info(f"instances remaining: {self.object_instances.shape[0]}")

    def assign_instances_to_segments(self):
        # assign instances to segments
        instance_ids = torch.ones_like(self.sg.frame_segs.ids) * -1

        for instance_idx in range(self.object_instances.shape[0]):
            instance_id = instance_idx + 1

            for frame_idx in range(self.object_instances.shape[1]):
                seg_mask = self.sg.frame_segs.ids == self.object_instances[instance_idx, frame_idx]
                instance_ids[seg_mask] = instance_id

        # merged child instances keep their segments under the surviving parent's id,
        # preserving coverage in frames where the parent has no segment of its own
        for child_row, parent_pos in self.merged_child_rows:
            instance_id = parent_pos + 1
            for frame_idx in range(child_row.shape[0]):
                if child_row[frame_idx] == 0:
                    continue
                seg_mask = self.sg.frame_segs.ids == child_row[frame_idx]
                instance_ids[seg_mask] = instance_id

        self.sg.frame_segs.instance_ids = instance_ids

    def assign_orphaned_parent_to_instance_by_union(self, unused_parent: torch.Tensor):
        """
        Compare the orphan, per frame, against the union of surface elements
        (mesh points or triangles) of each instance's segments visible in that
        frame. Per frame the orphan matches the instance with the largest shared
        element count at or above intersection_thresh; it is assigned only if all
        matching frames (including its own) agree on a single instance.

        Any instance with an above-threshold union overlap in any frame counts,
        so a second intersecting instance leaves the orphan unassigned.
        """
        sg = self.sg
        parent_id = int(unused_parent.item())

        elements = self.element_segments  # [N, (element_id, segment_id)]
        orphan_elements = elements[elements[:, 1] == parent_id][:, 0]
        if orphan_elements.numel() == 0:
            logger.info(f"orphaned parent {unused_parent} has no surface elements")
            return

        hits = elements[torch.isin(elements[:, 0], orphan_elements)]
        hits = hits[hits[:, 1] != parent_id]

        # map each intersecting segment to its instance and frame
        seg_ids = sg.frame_segs.ids.long()
        sorted_ids, order = torch.sort(seg_ids)
        hit_segment_ids = hits[:, 1].long().contiguous()
        pos = torch.searchsorted(sorted_ids, hit_segment_ids)
        pos = pos.clamp(max=sorted_ids.shape[0] - 1)
        valid = sorted_ids[pos] == hit_segment_ids
        hit_instances = sg.frame_segs.instance_ids[order[pos]]
        hit_frames = sg.frame_segs.frame_ids[order[pos]]
        keep = valid & (hit_instances > -1)

        if not keep.any():
            logger.info(f"no instance elements intersect orphaned parent {unused_parent}")
            return

        # exact union: an element covered by several segments of the same instance
        # in the same frame counts once
        triples = torch.stack(
            [hits[keep, 0].long(), hit_instances[keep].long(), hit_frames[keep].long()],
            dim=1,
        ).unique(dim=0)
        instance_frame_pairs, counts = torch.unique(triples[:, 1:], dim=0, return_counts=True)

        # Strict conflict handling is the deployed behavior: any second instance
        # with enough union support leaves the orphan unassigned.
        matched_instances = {
            instance_id
            for (instance_id, _frame_id), count in zip(instance_frame_pairs.tolist(), counts.tolist())
            if count >= self.intersection_thresh
        }

        if len(matched_instances) == 0:
            logger.info(f"no instance matches orphaned parent {unused_parent} with at least {self.intersection_thresh} elements")
            return

        if len(matched_instances) > 1:
            logger.info(f"orphaned parent {unused_parent} matches multiple instances {sorted(matched_instances)}; leaving unassigned")
            return

        best_instance_id = matched_instances.pop()
        sg.frame_segs.instance_ids[sg.frame_segs.ids == parent_id] = best_instance_id
        logger.info(f"assigned orphaned parent {unused_parent} to instance {best_instance_id} by union match")

    def assign_orphaned_parents_to_instances(self):
        possible_parents, counts = torch.unique(self.edges.children[:, 1], return_counts=True)
        parents = possible_parents[counts == 1]  # parents only appear one in children list as a child of themselves
        unused_parents = parents[~torch.isin(parents, self.object_instances.flatten())]

        # segments transferred from merged child instances already belong to an
        # instance and must not be treated as orphans
        assigned_seg_ids = self.sg.frame_segs.ids[self.sg.frame_segs.instance_ids > -1]
        unused_parents = unused_parents[~torch.isin(unused_parents, assigned_seg_ids)]

        # we want to work in order of size.
        unused_parent_sizes = self.sg.frame_segs.get_areas_by_ids(unused_parents)
        _, indicies = torch.sort(unused_parent_sizes, descending=True)
        sorted_unused_parents = unused_parents[indicies]

        for unused_parent in sorted_unused_parents:
            self.assign_orphaned_parent_to_instance_by_union(unused_parent)


def compute_segment_instance_ids_new(
    sg: FrameSegSceneGraph,
    *,
    intersection_thresh: int,
    assign_orphaned_parents: bool,
    edge_score_method: EdgeScoreMethod,
    element_segments: torch.Tensor,
) -> InstanceBuilder:
    builder = InstanceBuilder(
        sg,
        intersection_thresh,
        edge_score_method=edge_score_method,
        element_segments=element_segments,
    )

    builder.build_edges()

    builder.build_instance_candidates()

    # enforce each segment is in one instance_id
    builder.select_best_candidates()

    # remove instances that are children of other instances
    builder.merge_children()

    # assign instances to segments
    builder.assign_instances_to_segments()

    # assigned orphaned parents to the best instance (handle over segmentation)
    if assign_orphaned_parents:
        builder.assign_orphaned_parents_to_instances()

    return builder
