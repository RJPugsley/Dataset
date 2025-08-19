DEFINITIONS = {
    "bass-structured-groove-start": {
        "target": "first_downbeat",
        "stem": "bass",
        "position": "post",
        "conditions": {
            "min_post_energy": 0.002,
            "energy_jump": {
                "ratio": 3.0,
                "window_pre_beats": 32,
                "window_post_beats": 32
            },
            "pattern_detected": True,
            "pre_consistency": False,
            "pre_is_ambient": True
        },
        "style_tag": "groove-launch"
    },

    "bass-enters-post-loop": {
        "target": "loop_start",
        "stem": "bass",
        "position": "post",
        "conditions": {
            "min_post_energy": 0.003,
            "energy_jump": {
                "ratio": 4.0,
                "window_pre_beats": 32,
                "window_post_beats": 32
            },
            "pre_is_ambient": True,
            "pattern_detected": True
        },
        "style_tag": "delayed-bass-drop"
    },

    "staggered-genre-fill": {
        "target": "loop_start",
        "stem": "drums",
        "position": "pre",
        "conditions": {
            "hit_sequence": [
                {"type": "high_transient", "min_energy": 0.0025},
                {"type": "low_thump", "min_energy": 0.0045},
                {"type": "high_transient", "min_energy": 0.0025},
                {"type": "low_thump", "min_energy": 0.0045},
                {"type": "high_transient_group", "count_min": 2, "within_seconds": 1.5},
                {"type": "low_thump", "min_energy": 0.0045}
            ],
            "final_section_dropout": True
        },
        "style_tag": "bassline-percussive-fill"
    },

    "drum-entry-downbeat": {
        "target": "first_downbeat",
        "stem": "drums",
        "position": "post",
        "conditions": {
            "energy_jump": {
                "ratio": 3.0,
                "window_pre_beats": 32,
                "window_post_beats": 32
            },
            "min_post_energy": 0.005,
            "max_pre_peak": 0.0035,
            "pre_consistency": False,
            "post_consistency": True
        },
        "style_tag": "syncopated-kick-pattern"
    },

    "synth-ramp-intro": {
        "target": "first_downbeat",
        "stem": "other",
        "position": "pre",
        "conditions": {
            "energy_slope": "positive",
            "min_energy_range": 0.05,
            "max_pre_peak": 0.065,
            "pre_consistency": False,
            "post_consistency": False
        },
        "style_tag": "synth-ramp"
    },

    "synth-bed-post-downbeat": {
        "target": "first_downbeat",
        "stem": "other",
        "position": "post",
        "conditions": {
            "energy_range": {
                "min": 0.015,
                "max": 0.07
            },
            "pattern_detected": False,
            "post_consistency": True,
            "slope_variation": "low",
            "modulation_present": True,
            "dropouts": {
                "count_min": 2,
                "final_drop": True
            }
        },
        "style_tag": "chordal-gated-bed"
    },

    "synth-fill-loop-pre": {
        "target": "loop_start",
        "stem": "other",
        "position": "pre",
        "conditions": {
            "ramp_count": 2,
            "ramp_spacing": "even",
            "max_energy": 0.07,
            "min_energy": 0.02,
            "background_pad_present": True,
            "final_ramp_peak_near_loop": True
        },
        "style_tag": "atmospheric-synth-fill"
    },

    "synth-mute-post-loop": {
        "target": "loop_start",
        "stem": "other",
        "position": "post",
        "conditions": {
            "max_post_energy": 0.005,
            "pattern_detected": False,
            "post_consistency": False
        },
        "style_tag": "post-drop-silence"
    },

    "vocal-busy-pre-downbeat": {
        "target": "first_downbeat",
        "stem": "vocals",
        "position": "pre",
        "conditions": {
            "min_energy": 0.02,
            "max_energy": 0.08,
            "peak_density": "high",
            "reverb_tails": True
        },
        "style_tag": "female-vocal-reverbed-busy"
    },

    "vocal-busy-post-downbeat": {
        "target": "first_downbeat",
        "stem": "vocals",
        "position": "post",
        "conditions": {
            "min_energy": 0.03,
            "max_energy": 0.09,
            "peak_density": "very_high",
            "reverb_tails": True,
            "sustained_activity": True
        },
        "style_tag": "female-vocal-reverbed-busy"
    },

    "vocal-bridge-loop-pre": {
        "target": "loop_start",
        "stem": "vocals",
        "position": "pre",
        "conditions": {
            "initial_peak": True,
            "rapid_drop": True,
            "final_silence": True,
            "male_sample_present": True
        },
        "style_tag": "female-vocal-bridge"
    },

    "vocal-bleed-ignore-post-loop": {
        "target": "loop_start",
        "stem": "vocals",
        "position": "post",
        "conditions": {
            "max_post_energy": 0.006,
            "pattern_detected": False,
            "post_consistency": False
        },
        "style_tag": "bleed-ignore"
    }
}
