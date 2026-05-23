#include "AEConfig.h"
#include "AE_EffectVers.h"

resource 'PiPL' (16000) {
    {   /* array properties: 11 elements */
        /* [1] */
        Kind {
            AEEffect
        },
        /* [2] */
        Name {
            "Frame Echo"
        },
        /* [3] */
        Category {
            "Time-PlugIn"
        },
        /* [4] */
#ifdef AE_OS_WIN
        CodeWin64X86 {"EffectMain"},
#else
        CodeMacARM64 {"EffectMain"},
        CodeMacIntel64 {"EffectMain"},
#endif
        /* [5] */
        AE_PiPL_Version {
            2,
            0
        },
        /* [6] */
        AE_Effect_Spec_Version {
            PF_PLUG_IN_VERSION,
            PF_PLUG_IN_SUBVERS
        },
        /* [7] */
        AE_Effect_Version {
            0x80000
        },
        /* [8] */
        AE_Effect_Info_Flags {
            0
        },
        /* [9] */
        AE_Effect_Global_OutFlags {
            0x6000006
        },
        AE_Effect_Global_OutFlags_2 {
            0x08000100
        },
        /* [10] */
        AE_Effect_Match_Name {
            "Frame Echo"
        },
        /* [11] */
        AE_Reserved_Info {
            8
        }
    }
};