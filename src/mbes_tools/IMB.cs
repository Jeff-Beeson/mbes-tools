using System;
using System.Configuration;


namespace M3_TestSoftware
{
    public class IMBPacket
    {
        public const int HeaderSizeBytes = 56;
        public const int FooterSizeBytes = 44;

        public static byte[] SyncWord = { 0, 128, 0, 128, 0, 128, 0, 128 }; // Represents the sychronization sequence at the begining of a Packet Header

        public UInt32 dwVersion; 	// Version of this header
        public UInt32 dwSonarID; //Sonar identification
        public UInt32 DwSonarID //This is done in order to Bind the Data 
        {
            get { return dwSonarID; }
        }
        public UInt32[] dwSonarInfo = new UInt32[8];//Sonar information such as serial number, power up configurations
        public UInt32 dwTimeSec;  //Time stamp of current ping in seconds elapse since midnight (00:00:00), January 1, 1970
        public UInt32 DwTimeSec //This is done in order to Bind the Data 
        {
            get { return dwTimeSec; }
        }
        public UInt32 dwTimeMillisec; // Milliseconds part of current ping.
        public float fVelocitySound;    //Speed of sound in m/s
        public UInt32 nNumSamples;	//Total number of image samples.
        public float fNearRange;	//Minimum mode range in meters. No image below this range.
        public float fFarRange;    //Maximum mode range in meters.
        public float fSWST;        //Sampling Window Start Time in seconds
        public float fSWL;	        //Sampling Window Length
        public UInt16 nNumBeams;	//Total number of beams.
        public UInt16 wProcessingType;	            //	0: standard, 1: EIQ, 2: Profiling 3:Hybrid, 4:interferometry 5: Trawl 6: Profiling FFT
        public float[] BeamList = new float[1024]; // A list of angles for all beams up to the number of beams specified in nNumBeams field.
        public float fImageSampleInterval;	        //Image sample interval in seconds
        public UInt16 wImageDestination;           //Image designation, 0: main image window, n: zoom image window n, n<=4
        public UInt16 wReserved2;	    //Reserved field
        public UInt32 dwModeID;        //Unique mode ID, application ID
        public Int32 nNumHybridPRI; 	//Number of PRIs in a hybrid mode. 1 if not in hybrid mode.
        public Int32 nHybridIndex;		//0 based index used in hybrid mode up to (nNumHybridPRI -1)
        public UInt16 nPhaseSeqLength;	//Spatial phase sequence length
        public UInt16 iPhaseSeqIndex;	//Spatial phase sequence length index 0..(nPhaseSeqLength – 1)
        public UInt16 nNumImages;      //Number sub-images for one PRI
        public UInt16 iSubImageIndex;	//Sub-image index 0 .. (nNumImages – 1)
        public UInt32 dwSonarFreq; 	//Sonar frequency in Hz.
        public UInt32 dwPulseLength;	//Transmit pulse length in microseconds.
        public UInt32 dwPingNumber;	//Ping counter. Rolls back to zero if reaches 0xFFFFFFFF       
        public UInt32 DwPingNumber //This is done in order to Bind the Data 
        {
            get { return dwPingNumber; }
        }
        public float fRXFilterBW;	                    //RX filter bandwidth in Hz
        public float fRXNominalResolution;              //RX nominal resolution in metres
        public float fPulseRepFreq;	                    //Pulse Repetition Frequency in Hz
        public char[] strAppName = new char[128];	    //Application name, null terminated. Maximum 128 characters.
        public char[] strTXPulseName64 = new char[64];	//TX pulse name, null terminated. Maximum 64 characters.
        public TVG_PARAMS sTVGParameters = new TVG_PARAMS();	//TVG Parameters: (Available if dwVersion >=1) 12
        public float fCompassHeading;          //heading of current ping in decimal degrees
        public float MagneticVariation;	       //Magnetic variation in decimal degrees
        public float fPitch;        //Pitch in decimal degrees
        public float fRoll;	        //Roll in decimal degrees
        public float fDepth;	    //Depth in decimal meters
        public float Temperature;	//Temperature in decimal Celsius
        public float fXOffset;	    //Translational offset in X axis
        public float fYOffset;	    //Translational offset in Y axis
        public float fZOffset;	    //Translational offset in Z axis
        public float fXRotOffset;   //Rotational offset about X axis(Pitch offset).
        public float fYRotOffset;	//Rotational offset about Y axis(Roll offset)
        public float fZRotOffset;	//Rotational offset about Z axis(Yaw offset)
        public UInt32 dwMounting;	//Mounting orientation: (If dwVersion < 5  0: mounting Down or Aft 1: mounting Up or Fore) (If dwVersion >=5  0: Forward 1: Forward Inverted 2: Roll Right 3: Roll Left 4: Upward 5: Upward Inverted 6: Downward 7: Downward Inverted) (Deploy Config Head Mounting Parameter. For information only, mounting orientation is fully described by the fXRotOffset, fYRotOffset and fYRotOffset fields.)
        public double dbLatitude;	//Latitude of current ping in decimal degrees   
        public double dbLongitude;	//Longitude of current ping in decimal degrees
        public float fTXWST;	    //TX Window Start Time in seconds. Available if (dwVersion >= 4)
        public byte bHeadSensorVersion;	//Head Sensor Version
        public byte HeadHWStatus;	//Head hardware status: 0: Normal 1: High temperature warning 2: High temperature shutdown
        public byte byReserved1;	//Reserved field
        public byte byReserved2;	//Reserved field
        public float fInternalSensorHeading;	//Heading from internal sensor
        public float fInternalSensorPitch;	    //Pitch from internal sensor
        public float fInternalSensorRoll;	    //Roll from internal sensor
        public M3_ROTATOR_OFFSETS[] sAxesRotatorOffsets = new M3_ROTATOR_OFFSETS[3]//Rotator offset for a rotator mounted M3 system 48
            { new M3_ROTATOR_OFFSETS(), new M3_ROTATOR_OFFSETS(), new M3_ROTATOR_OFFSETS() };
        public UInt16 nStartElement;		    //Used if nNumImages > 1. Zero based start element of the sub array for the current sub image.
        public UInt16 nEndElement;	            //Used if nNumImages > 1. Zero based end element of the sub array for the current sub image.
        public char[] strCustomText1 = new char[32];	//Custom text field 1
        public char[] strCustomText2 = new char[32];	//Custom text field 2
        public float fLocalTimeOffset;         //Local time zone offset in decimal hours relative to UTC.
        public float fVesselSOG;	            //Vessel Speed Over Ground in knots
        public float fHeave;                   //Heave in meters.
        public float fPRIMinRange;	            //Minimum PRI range in meters.
        public float fPRIMaxRange;	            //Maximum PRI range in meters.
        public float fSoundSpeed_RT;	        //Real-time sound speed from SV sensors.
        public byte[] Reserved = new byte[3856];
        public float[,] fPacketBody;
    }
    public class M3_ROTATOR_OFFSETS
    {
        public float fOffsetA; //Offset A in meters
        public float fOffsetB; //Offset B in meters
        public float fOffsetR; //Offset R in meters
        public float fRotAngle; //Rotator Angle in degrees
    }
    public class TVG_PARAMS
    {
        public UInt16 SpreadingCoeff; //Factor A, the spreading coefficient
        public UInt16 AbsorptionCoeff; //FactorB, the absorption coefficient in dB/km
        public float fTVGCruve; //Factor C, the TVG curve offset in dB
        public float fMaxGain; //Factor L, the maximum gain limit in dB
    }
}

