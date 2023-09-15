from typing import List, Union


def get_outputs(diarization, model, request_id_input):
    outputs = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # start of turn
        outputs.append(turn.start)
        # end of turn
        outputs.append(turn.end)
        # speaker id
        outputs.append(speaker)
        model.logger.log_verbose(
            f"{model.instance}:{request_id_input}: "
            + f"{round(turn.start, 2)} {round(turn.end, 2)} {speaker}"
        )

    outputs = rename_speakers_in_appearance_order(outputs)

    return outputs


def rename_speakers_in_appearance_order(
    diarization_as_list: List[Union[str, float]]
):
    """
    inputs:
      [0, 1, 'SPEAKER_02', 1, 2, 'SPEAKER_00', 2, 3, 'SPEAKER_01', 3, 4, 'SPEAKER_02']  # noqa E501
    outputs:
      [0, 1, 'SPEAKER_00', 1, 2, 'SPEAKER_01', 2, 3, 'SPEAKER_02', 3, 4, 'SPEAKER_00']  # noqa E501
    """
    speaker_map = {}
    speaker_count = 0
    # Loop on speakers only
    for i in range(2, len(diarization_as_list), 3):
        speaker = diarization_as_list[i]
        # Map the speaker new name if it's not
        if speaker not in speaker_map:
            speaker_map[speaker] = f"SPEAKER_{'{:02d}'.format(speaker_count)}"
            speaker_count += 1
        diarization_as_list[i] = speaker_map[speaker]
    return diarization_as_list
