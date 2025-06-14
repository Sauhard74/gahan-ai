def collate_sequences(batch):
    """
    Custom collate function for sequence data.
    Handles variable number of objects per frame.
    """
    try:
        # Separate components
        sequences = []
        targets = []
        metadata = []
        
        for item in batch:
            if item is None:
                print("⚠️ Skipping None item in batch")
                continue
                
            sequences.append(item['sequence'])
            targets.append(item['targets'])
            metadata.append(item['metadata'])
        
        if len(sequences) == 0:
            print("❌ No valid sequences in batch")
            return None
        
        # Stack sequences (B, T, C, H, W)
        sequences = torch.stack(sequences)
        
        # Process targets - pad to same number of objects
        max_objects = max(len(target['boxes']) for target in targets)
        
        if max_objects == 0:
            print("⚠️ No objects found in any target")
            # Create dummy targets
            batch_size = len(targets)
            padded_targets = {
                'boxes': torch.zeros(batch_size, 1, 4),
                'labels': torch.zeros(batch_size, 1, dtype=torch.long),
                'cutting_behavior': torch.zeros(batch_size, dtype=torch.float32)
            }
        else:
            # Pad targets to same size
            padded_boxes = []
            padded_labels = []
            cutting_behaviors = []
            
            for target in targets:
                boxes = target['boxes']
                labels = target['labels']
                cutting = target['cutting_behavior']
                
                # Pad boxes and labels
                if len(boxes) < max_objects:
                    pad_size = max_objects - len(boxes)
                    # Pad with zeros (background)
                    boxes = torch.cat([boxes, torch.zeros(pad_size, 4)], dim=0)
                    labels = torch.cat([labels, torch.zeros(pad_size, dtype=torch.long)], dim=0)
                
                padded_boxes.append(boxes)
                padded_labels.append(labels)
                cutting_behaviors.append(cutting)
            
            padded_targets = {
                'boxes': torch.stack(padded_boxes),
                'labels': torch.stack(padded_labels),
                'cutting_behavior': torch.stack(cutting_behaviors)
            }
        
        return {
            'sequences': sequences,
            'targets': padded_targets,
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"❌ Error in collate_sequences: {e}")
        import traceback
        traceback.print_exc()
        return None 