import json
import datetime

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)
    
def naive_data_collator(batch):
    """Does nothing, only for dataloader to batch samples 
    and not to convert them to tensors
    
    batch (list): list of dicts 
    Returns:
        list: list of dicts
    """    
    return batch

def format_duration(seconds):
    """Format seconds into nice output

    Args:
        seconds (int): seconds

    Returns:
        str: formatted string
    """    
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}h:{int(minutes):02}m:{int(seconds):02}s"


def validate_last_json(jsonl_file):
    with open(jsonl_file, 'rb') as f:
        f.seek(0, 2)  # Move to the end of the file
        end_pos = f.tell()
        
        # Read backward until we find a newline
        buffer = b''
        for pos in range(end_pos - 1, -1, -1):
            f.seek(pos)
            char = f.read(1)
            if char == b'\n' and buffer:  # Found the end of the last JSON
                break
            buffer = char + buffer
        
        # Decode and validate JSON
        try:
            last_json = json.loads(buffer.decode('utf-8'))
            print("Last JSON is valid:", last_json)
            return True
        except json.JSONDecodeError as e:
            print("Last JSON is invalid:", e)
            return False
