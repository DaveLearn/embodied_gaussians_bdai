import torch
from frame_seg_init.mask_stack import MaskStack


def test_mask_stack_starts_empty():
    mask_stack = MaskStack()
    assert mask_stack.masks.shape[0] == 0


def test_mask_stack_sets_dims_on_first_add():
    mask_stack = MaskStack()
    mask_stack.add(1, torch.zeros(10, 10, dtype=torch.bool))
    assert mask_stack.masks.shape == (1, 10, 10)


def test_mask_stack_can_add_and_retrieve_single_mask():
    mask_stack = MaskStack()
    mask = torch.zeros(10, 10, dtype=torch.bool)
    mask[2:4, 2:4] = True
    mask_stack.add(1, mask)

    assert torch.equal(mask_stack.mask_for_id(1).cpu(), mask)


def test_mask_stack_can_add_and_retrieve_multiple_non_overlapping_masks():
    mask_stack = MaskStack()
    mask1 = torch.zeros(10, 10, dtype=torch.bool)
    mask1[2:4, 2:4] = True
    mask_stack.add(1, mask1)

    mask2 = torch.zeros(10, 10, dtype=torch.bool)
    mask2[5:7, 5:7] = True
    mask_stack.add(2, mask2)

    # we can retrieve the masks
    assert torch.equal(mask_stack.mask_for_id(1).cpu(), mask1)
    assert torch.equal(mask_stack.mask_for_id(2).cpu(), mask2)

    # we only used a single layer
    assert mask_stack.depth() == 1


def test_mask_stack_can_add_and_retrieve_overlapping_masks():
    mask_stack = MaskStack()
    mask1 = torch.zeros(10, 10, dtype=torch.bool)
    mask1[2:4, 2:4] = True
    mask_stack.add(1, mask1)

    mask2 = torch.zeros(10, 10, dtype=torch.bool)
    mask2[3:5, 3:5] = True
    mask_stack.add(2, mask2)

    # we can retrieve the masks
    assert torch.equal(mask_stack.mask_for_id(1).cpu(), mask1)
    assert torch.equal(mask_stack.mask_for_id(2).cpu(), mask2)

    # we needed 2 layers
    assert mask_stack.depth() == 2


def test_mask_stack_get_ids_and_areas():
    mask_stack = MaskStack()
    mask1 = torch.zeros(10, 10, dtype=torch.bool)
    mask1[2:4, 2:4] = True
    mask_stack.add(1, mask1)

    mask2 = torch.zeros(10, 10, dtype=torch.bool)
    mask2[3:5, 3:5] = True
    mask_stack.add(2, mask2)

    ids, areas = mask_stack.get_ids_and_areas()

    assert torch.equal(ids.cpu(), torch.tensor([1, 2]))
    assert torch.equal(areas.cpu(), torch.tensor([4, 4]))


def test_mask_stack_can_fit_in_layer():
    mask_stack = MaskStack()
    mask1 = torch.zeros(10, 10, dtype=torch.bool)
    mask1[2:4, 2:4] = True
    mask_stack.add(1, mask1)

    mask2 = torch.zeros(10, 10, dtype=torch.bool)
    mask2[6:8, 6:8] = True
    mask_stack.add(2, mask2)

    mask3 = torch.zeros(10, 10, dtype=torch.bool)
    mask3[3:5, 3:5] = True

    assert not mask_stack.can_fit_in_layer(mask3, 0)
    assert mask_stack.can_fit_in_layer(mask3, 1)


def test_mask_stack_insert_into_layer():
    mask_stack = MaskStack()
    mask1 = torch.zeros(10, 10, dtype=torch.bool)
    mask1[2:4, 2:4] = True
    mask_stack.add(1, mask1)

    mask2 = torch.zeros(10, 10, dtype=torch.bool)
    mask2[3:8, 3:8] = True
    mask_stack.add(2, mask2)

    assert mask_stack.depth() == 2

    mask3 = torch.zeros(10, 10, dtype=torch.bool)
    mask3[3:5, 3:5] = True

    mask_stack.insert_into_layer(3, mask3, layer_idx=0)
    assert mask_stack.depth() == 2
    assert torch.equal(mask_stack.mask_for_id(3).cpu(), mask3)

    mask_stack.insert_into_layer(4, mask3, 1)
    assert mask_stack.depth() == 2
    assert torch.equal(mask_stack.mask_for_id(4).cpu(), mask3)

    # we can add to a new layer
    mask_stack.insert_into_layer(5, mask3, 2)
    assert mask_stack.depth() == 3
    assert torch.equal(mask_stack.mask_for_id(5).cpu(), mask3)


def test_mask_stack_add_respects_overlap():
    mask_stack = MaskStack(allowed_overlap_fraction=0.5)
    mask1 = torch.zeros(10, 10, dtype=torch.bool)
    mask1[2:4, 2:4] = True
    mask_stack.add(1, mask1)

    mask2 = torch.zeros(10, 10, dtype=torch.bool)
    mask2[6:8, 6:8] = True
    mask_stack.add(2, mask2)

    # add a mask that overlaps with mask1 by 1 pixel (frac of 0.25)
    mask3 = torch.zeros(10, 10, dtype=torch.bool)
    mask3[3:5, 3:5] = True
    mask_stack.add(3, mask3)

    assert mask_stack.depth() == 1

    # if we try to add it again (overlap of 1.0) we end up in second layer
    mask_stack.add(4, mask3)
    assert mask_stack.depth() == 2


def test_mask_stack_import_export_layer_as_image():
    mask_stack = MaskStack(allowed_overlap_fraction=0)
    mask1 = torch.zeros(10, 10, dtype=torch.bool)
    mask1[2:4, 2:4] = True
    mask_stack.add(1, mask1)

    mask2 = torch.zeros(10, 10, dtype=torch.bool)
    mask2[6:8, 6:8] = True
    mask_stack.add(2, mask2)

    # add a mask that overlaps with mask1 by 1 pixel (frac of 0.25)
    mask3 = torch.zeros(10, 10, dtype=torch.bool)
    mask3[3:5, 3:5] = True
    mask_stack.add(3, mask3)

    assert mask_stack.depth() == 2
    layer0_img = mask_stack.export_layer_as_image(0)
    layer1_img = mask_stack.export_layer_as_image(1)

    # test by import
    imported_stack = MaskStack(allowed_overlap_fraction=0)
    imported_stack.update_layer_from_image(0, layer0_img)
    imported_stack.update_layer_from_image(1, layer1_img)

    assert torch.equal(imported_stack.mask_for_id(1).cpu(), mask1)
    assert torch.equal(imported_stack.mask_for_id(2).cpu(), mask2)
    assert torch.equal(imported_stack.mask_for_id(3).cpu(), mask3)

    assert torch.equal(imported_stack.masks, mask_stack.masks)


def test_mask_stack_save_load_from_file():
    mask_stack = MaskStack(allowed_overlap_fraction=0)
    mask1 = torch.zeros(10, 10, dtype=torch.bool)
    mask1[2:4, 2:4] = True
    mask_stack.add(1, mask1)

    mask2 = torch.zeros(10, 10, dtype=torch.bool)
    mask2[6:8, 6:8] = True
    mask_stack.add(2, mask2)

    # add a mask that overlaps with mask1 by 1 pixel (frac of 0.25)
    mask3 = torch.zeros(10, 10, dtype=torch.bool)
    mask3[3:5, 3:5] = True
    mask_stack.add(3, mask3)

    from tempfile import TemporaryDirectory
    import os

    # get a temporary directory to save the file using the operating systems
    with TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test.ms")
        mask_stack.save_to_file(filename)

        # load the file
        loaded_stack = MaskStack.from_file(filename)
        assert torch.equal(loaded_stack.masks, mask_stack.masks)
