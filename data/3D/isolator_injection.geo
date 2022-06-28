SetFactory("OpenCASCADE");

If(Exists(size))
    basesize=size;
Else
    basesize=0.002;
EndIf

If(Exists(blratio))
    boundratio=blratio;
Else
    boundratio=2.0;
EndIf

If(Exists(blratiocavity))
    boundratiocavity=blratiocavity;
Else
    boundratiocavity=2.0;
EndIf

If(Exists(blratioinjector))
    boundratioinjector=blratioinjector;
Else
    boundratioinjector=2.0;
EndIf

If(Exists(injectorfac))
    injector_factor=injectorfac;
Else
    injector_factor=10.0;
EndIf

If(Exists(shearfac))
    shear_factor=shearfac;
Else
    shear_factor=1.0;
EndIf

If(Exists(cavityfac))
    cavity_factor=cavityfac;
Else
    cavity_factor=1.0;
EndIf

bigsize = basesize*4;     // the biggest mesh size 
inletsize = basesize*2;   // background mesh size upstream of the nozzle
isosize = basesize;       // background mesh size in the isolator
nozzlesize = basesize/2;       // background mesh size in the isolator
cavitysize = basesize/cavity_factor; // background mesh size in the cavity region
shearsize = cavitysize/shear_factor;

inj_h=4.e-3;  // height of injector (bottom) from floor
inj_t=1.59e-3; // diameter of injector
inj_d = 20e-3; // length of injector
injectorsize = inj_t/injector_factor; // background mesh size in the cavity region

Printf("basesize = %f", basesize);
Printf("inletsize = %f", inletsize);
Printf("isosize = %f", isosize);
Printf("nozzlesize = %f", nozzlesize);
Printf("cavitysize = %f", cavitysize);
Printf("shearsize = %f", shearsize);
Printf("injectorsize = %f", injectorsize);
Printf("boundratio = %f", boundratio);
Printf("boundratiocavity = %f", boundratiocavity);
Printf("boundratioinjector = %f", boundratioinjector);

//Top Wall Back
Point(1) = {0.21, 0.0270645, 0.0, basesize};
Point(2) = {0.2280392417062, 0.0270645, 0.0, basesize};
Point(3) = {0.2287784834123, 0.0270645, 0.0, basesize};
Point(4) = {0.2295177251185, 0.0270645, 0.0, basesize};
Point(5) = {0.2302569668246, 0.0270645, 0.0, basesize};
Point(6) = {0.2309962085308, 0.0270645, 0.0, basesize};
Point(7) = {0.231735450237, 0.0270645, 0.0, basesize};
Point(8) = {0.2324746919431, 0.0270645, 0.0, basesize};
Point(9) = {0.2332139336493, 0.0270645, 0.0, basesize};
Point(10) = {0.2339531753555, 0.0270645, 0.0, basesize};
Point(11) = {0.2346924170616, 0.02679523462424, 0.0, basesize};
Point(12) = {0.2354316587678, 0.02628798808666, 0.0, basesize};
Point(13) = {0.2361709004739, 0.02578074154909, 0.0, basesize};
Point(14) = {0.2369101421801, 0.02527349501151, 0.0, basesize};
Point(15) = {0.2376493838863, 0.02476624847393, 0.0, basesize};
Point(16) = {0.2383886255924, 0.02425900193636, 0.0, basesize};
Point(17) = {0.2391278672986, 0.02375175539878, 0.0, basesize};
Point(18) = {0.2398671090047, 0.02324450886121, 0.0, basesize};
Point(19) = {0.2406063507109, 0.02273726232363, 0.0, basesize};
Point(20) = {0.2413455924171, 0.02223001578605, 0.0, basesize};
Point(21) = {0.2420848341232, 0.02172276924848, 0.0, basesize};
Point(22) = {0.2428240758294, 0.0212155227109, 0.0, basesize};
Point(23) = {0.2435633175355, 0.02070827617332, 0.0, basesize};
Point(24) = {0.2443025592417, 0.02020102963575, 0.0, basesize};
Point(25) = {0.2450418009479, 0.01969378309817, 0.0, basesize};
Point(26) = {0.245781042654, 0.0191865365606, 0.0, basesize};
Point(27) = {0.2465202843602, 0.01867929002302, 0.0, basesize};
Point(28) = {0.2472595260664, 0.01817204348544, 0.0, basesize};
Point(29) = {0.2479987677725, 0.01766479694787, 0.0, basesize};
Point(30) = {0.2487380094787, 0.01715755041029, 0.0, basesize};
Point(31) = {0.2494772511848, 0.01665030387271, 0.0, basesize};
Point(32) = {0.250216492891, 0.01614305733514, 0.0, basesize};
Point(33) = {0.2509557345972, 0.01563581079756, 0.0, basesize};
Point(34) = {0.2516949763033, 0.01512856425999, 0.0, basesize};
Point(35) = {0.2524342180095, 0.01462131772241, 0.0, basesize};
Point(36) = {0.2531734597156, 0.01411407118483, 0.0, basesize};
Point(37) = {0.2539127014218, 0.01360682464726, 0.0, basesize};
Point(38) = {0.254651943128, 0.01309957810968, 0.0, basesize};
Point(39) = {0.2553911848341, 0.01259233157211, 0.0, basesize};
Point(40) = {0.2561304265403, 0.01208508503453, 0.0, basesize};
Point(41) = {0.2568696682464, 0.01157783849695, 0.0, basesize};
Point(42) = {0.2576089099526, 0.01107059195938, 0.0, basesize};
Point(43) = {0.2583481516588, 0.0105633454218, 0.0, basesize};
Point(44) = {0.2590873933649, 0.01005609888422, 0.0, basesize};
Point(45) = {0.2598266350711, 0.009548852346649, 0.0, basesize};
Point(46) = {0.2605658767773, 0.009041605809072, 0.0, basesize};
Point(47) = {0.2613051184834, 0.008534359271496, 0.0, basesize};
Point(48) = {0.2620443601896, 0.00802711273392, 0.0, basesize};
Point(49) = {0.2627836018957, 0.007519866196344, 0.0, basesize};
Point(50) = {0.2635228436019, 0.007012619658768, 0.0, basesize};
Point(51) = {0.2642620853081, 0.006505373121192, 0.0, basesize};
Point(52) = {0.2650013270142, 0.005998126583615, 0.0, basesize};
Point(53) = {0.2657405687204, 0.005490880046039, 0.0, basesize};
Point(54) = {0.2664798104265, 0.004983633508463, 0.0, basesize};
Point(55) = {0.2672190521327, 0.004476386970887, 0.0, basesize};
Point(56) = {0.2679582938389, 0.003969140433311, 0.0, basesize};
Point(57) = {0.268697535545, 0.003461893895735, 0.0, basesize};
Point(58) = {0.2694367772512, 0.002954647358158, 0.0, basesize};
Point(59) = {0.2701760189573, 0.002447400820582, 0.0, basesize};
Point(60) = {0.2709152606635, 0.001940154283006, 0.0, basesize};
Point(61) = {0.2716545023697, 0.00143290774543, 0.0, basesize};
Point(62) = {0.2723937440758, 0.0009256612078538, 0.0, basesize};
Point(63) = {0.273132985782, 0.0004184146702776, 0.0, basesize};
Point(64) = {0.2738722274882, -0.00008883186729857, 0.0, basesize};
Point(65) = {0.2746114691943, -0.0005960784048747, 0.0, basesize};
Point(66) = {0.2753507109005, -0.001103324942451, 0.0, basesize};
Point(67) = {0.2760899526066, -0.001610571480027, 0.0, basesize};
Point(68) = {0.2768291943128, -0.0021178180176, 0.0, basesize};
Point(69) = {0.277568436019, -0.002625063418531, 0.0, basesize};
Point(70) = {0.2783076777251, -0.003128071371827, 0.0, basesize};
Point(71) = {0.2790469194313, -0.00356543025825, 0.0, basesize};
Point(72) = {0.2797861611374, -0.003924485596916, 0.0, basesize};
Point(73) = {0.2805254028436, -0.004209800511799, 0.0, basesize};
Point(74) = {0.2812646445498, -0.004425962626834, 0.0, basesize};
Point(75) = {0.2820038862559, -0.004577559566121, 0.0, basesize};
Point(76) = {0.2827431279621, -0.004669178953759, 0.0, basesize};
Point(77) = {0.2834823696682, -0.004705408413847, 0.0, basesize};
Point(78) = {0.2842216113744, -0.004697204954745, 0.0, basesize};
Point(79) = {0.2849608530806, -0.00465704436755, 0.0, basesize};
Point(80) = {0.2857000947867, -0.004586244418798, 0.0, basesize};
Point(81) = {0.2864393364929, -0.004485025473862, 0.0, basesize};
Point(82) = {0.2871785781991, -0.004353607898117, 0.0, basesize};
Point(83) = {0.2879178199052, -0.004192212056935, 0.0, basesize};
Point(84) = {0.2886570616114, -0.00400105831569, 0.0, basesize};
Point(85) = {0.2893963033175, -0.003780367039754, 0.0, basesize};
Point(86) = {0.2901355450237, -0.003530358594502, 0.0, basesize};
Point(87) = {0.2908747867299, -0.003251253345306, 0.0, basesize};
Point(88) = {0.291614028436, -0.002943271657539, 0.0, basesize};
Point(89) = {0.2923532701422, -0.002613060084159, 0.0, basesize};
Point(90) = {0.2930925118483, -0.00228623916318, 0.0, basesize};
Point(91) = {0.2938317535545, -0.001965379671836, 0.0, basesize};
Point(92) = {0.2945709952607, -0.001650408524638, 0.0, basesize};
Point(93) = {0.2953102369668, -0.001341252636095, 0.0, basesize};
Point(94) = {0.296049478673, -0.001037838920719, 0.0, basesize};
Point(95) = {0.2967887203791, -0.0007400942930211, 0.0, basesize};
Point(96) = {0.2975279620853, -0.0004479456675107, 0.0, basesize};
Point(97) = {0.2982672037915, -0.0001613199586989, 0.0, basesize};
Point(98) = {0.2990064454976, 0.0001198559189035, 0.0, basesize};
Point(99) = {0.2997456872038, 0.0003956550507858, 0.0, basesize};
Point(100) = {0.30048492891, 0.0006661505224375, 0.0, basesize};
Point(101) = {0.3012241706161, 0.0009314154193479, 0.0, basesize};
Point(102) = {0.3019634123223, 0.001191522827006, 0.0, basesize};
Point(103) = {0.3027026540284, 0.001446545830902, 0.0, basesize};
Point(104) = {0.3034418957346, 0.001696557516524, 0.0, basesize};
Point(105) = {0.3041811374408, 0.001941630969363, 0.0, basesize};
Point(106) = {0.3049203791469, 0.002181839274906, 0.0, basesize};
Point(107) = {0.3056596208531, 0.002417255518644, 0.0, basesize};
Point(108) = {0.3063988625592, 0.002647952786067, 0.0, basesize};
Point(109) = {0.3071381042654, 0.002874004162662, 0.0, basesize};
Point(110) = {0.3078773459716, 0.003095482733921, 0.0, basesize};
Point(111) = {0.3086165876777, 0.003312461585331, 0.0, basesize};
Point(112) = {0.3093558293839, 0.003525013802383, 0.0, basesize};
Point(113) = {0.31009507109, 0.003733212470565, 0.0, basesize};
Point(114) = {0.3108343127962, 0.003937130675367, 0.0, basesize};
Point(115) = {0.3115735545024, 0.004136841502279, 0.0, basesize};
Point(116) = {0.3123127962085, 0.004332418036789, 0.0, basesize};
Point(117) = {0.3130520379147, 0.004523933364387, 0.0, basesize};
Point(118) = {0.3137912796209, 0.004711460570563, 0.0, basesize};
Point(119) = {0.314530521327, 0.004895072740805, 0.0, basesize};
Point(120) = {0.3152697630332, 0.005074842960603, 0.0, basesize};
Point(121) = {0.3160090047393, 0.005250844315447, 0.0, basesize};
Point(122) = {0.3167482464455, 0.005423149890825, 0.0, basesize};
Point(123) = {0.3174874881517, 0.005591832772228, 0.0, basesize};
Point(124) = {0.3182267298578, 0.005756966045143, 0.0, basesize};
Point(125) = {0.318965971564, 0.005918622795062, 0.0, basesize};
Point(126) = {0.3197052132701, 0.006076876107472, 0.0, basesize};
Point(127) = {0.3204444549763, 0.006231799067864, 0.0, basesize};
Point(128) = {0.3211836966825, 0.006383464761726, 0.0, basesize};
Point(129) = {0.3219229383886, 0.006531946274548, 0.0, basesize};
Point(130) = {0.3226621800948, 0.00667731669182, 0.0, basesize};
Point(131) = {0.3234014218009, 0.00681964909903, 0.0, basesize};
Point(132) = {0.3241406635071, 0.006959016581669, 0.0, basesize};
Point(133) = {0.3248799052133, 0.007095492225225, 0.0, basesize};
Point(134) = {0.3256191469194, 0.007229149115187, 0.0, basesize};
Point(135) = {0.3263583886256, 0.007360060337045, 0.0, basesize};
Point(136) = {0.3270976303318, 0.007488298976289, 0.0, basesize};
Point(137) = {0.3278368720379, 0.007613938118408, 0.0, basesize};
Point(138) = {0.3285761137441, 0.00773705084889, 0.0, basesize};
Point(139) = {0.3293153554502, 0.007857710253226, 0.0, basesize};
Point(140) = {0.3300545971564, 0.007975989416904, 0.0, basesize};
Point(141) = {0.3307938388626, 0.008091961425415, 0.0, basesize};
Point(142) = {0.3315330805687, 0.008205699364247, 0.0, basesize};
Point(143) = {0.3322723222749, 0.008317276318889, 0.0, basesize};
Point(144) = {0.333011563981, 0.008426765374832, 0.0, basesize};
Point(145) = {0.3337508056872, 0.008534231284918, 0.0, basesize};
Point(146) = {0.3344900473934, 0.008639591517526, 0.0, basesize};
Point(147) = {0.3352292890995, 0.008742817528548, 0.0, basesize};
Point(148) = {0.3359685308057, 0.008843926036179, 0.0, basesize};
Point(149) = {0.3367077725118, 0.008942933758618, 0.0, basesize};
Point(150) = {0.337447014218, 0.009039857414062, 0.0, basesize};
Point(151) = {0.3381862559242, 0.00913471372071, 0.0, basesize};
Point(152) = {0.3389254976303, 0.009227519396758, 0.0, basesize};
Point(153) = {0.3396647393365, 0.009318291160404, 0.0, basesize};
Point(154) = {0.3404039810427, 0.009407045729847, 0.0, basesize};
Point(155) = {0.3411432227488, 0.009493799823283, 0.0, basesize};
Point(156) = {0.341882464455, 0.00957857015891, 0.0, basesize};
Point(157) = {0.3426217061611, 0.009661373454925, 0.0, basesize};
Point(158) = {0.3433609478673, 0.009742226429528, 0.0, basesize};
Point(159) = {0.3441001895735, 0.009821145800914, 0.0, basesize};
Point(160) = {0.3448394312796, 0.009898148287282, 0.0, basesize};
Point(161) = {0.3455786729858, 0.00997325060683, 0.0, basesize};
Point(162) = {0.3463179146919, 0.01004646947775, 0.0, basesize};
Point(163) = {0.3470571563981, 0.01011782161825, 0.0, basesize};
Point(164) = {0.3477963981043, 0.01018732374652, 0.0, basesize};
Point(165) = {0.3485356398104, 0.01025499258077, 0.0, basesize};
Point(166) = {0.3492748815166, 0.01032084483917, 0.0, basesize};
Point(167) = {0.3500141232227, 0.01038489723995, 0.0, basesize};
Point(168) = {0.3507533649289, 0.01044716650128, 0.0, basesize};
Point(169) = {0.3514926066351, 0.01050766934138, 0.0, basesize};
Point(170) = {0.3522318483412, 0.01056642247844, 0.0, basesize};
Point(171) = {0.3529710900474, 0.01062344263065, 0.0, basesize};
Point(172) = {0.3537103317536, 0.01067874651621, 0.0, basesize};
Point(173) = {0.3544495734597, 0.01073235085333, 0.0, basesize};
Point(174) = {0.3551888151659, 0.01078427236019, 0.0, basesize};
Point(175) = {0.355928056872, 0.010834527755, 0.0, basesize};
Point(176) = {0.3566672985782, 0.01088313375595, 0.0, basesize};
Point(177) = {0.3574065402844, 0.01093010708125, 0.0, basesize};
Point(178) = {0.3581457819905, 0.01097546444908, 0.0, basesize};
Point(179) = {0.3588850236967, 0.01101922257765, 0.0, basesize};
Point(180) = {0.3596242654028, 0.01106139818516, 0.0, basesize};
Point(181) = {0.360363507109, 0.0111020079898, 0.0, basesize};
Point(182) = {0.3611027488152, 0.01114106870976, 0.0, basesize};
Point(183) = {0.3618419905213, 0.01117859706326, 0.0, basesize};
Point(184) = {0.3625812322275, 0.01121460976848, 0.0, basesize};
Point(185) = {0.3633204739336, 0.01124912354362, 0.0, basesize};
Point(186) = {0.3640597156398, 0.01128215510688, 0.0, basesize};
Point(187) = {0.364798957346, 0.01131372117646, 0.0, basesize};
Point(188) = {0.3655381990521, 0.01134383847056, 0.0, basesize};
Point(189) = {0.3662774407583, 0.01137252370737, 0.0, basesize};
Point(190) = {0.3670166824645, 0.01139979360508, 0.0, basesize};
Point(191) = {0.3677559241706, 0.01142566488191, 0.0, basesize};
Point(192) = {0.3684951658768, 0.01145015425605, 0.0, basesize};
Point(193) = {0.3692344075829, 0.01147327844568, 0.0, basesize};
Point(194) = {0.3699736492891, 0.01149505416902, 0.0, basesize};
Point(195) = {0.3707128909953, 0.01151549814426, 0.0, basesize};
Point(196) = {0.3714521327014, 0.01153462708959, 0.0, basesize};
Point(197) = {0.3721913744076, 0.01155245772322, 0.0, basesize};
Point(198) = {0.3729306161137, 0.01156900676334, 0.0, basesize};
Point(199) = {0.3736698578199, 0.01158429092815, 0.0, basesize};
Point(200) = {0.3744090995261, 0.01159832693585, 0.0, basesize};
Point(201) = {0.3751483412322, 0.01161113150463, 0.0, basesize};
Point(202) = {0.3758875829384, 0.01162272135269, 0.0, basesize};
Point(203) = {0.3766268246445, 0.01163311319823, 0.0, basesize};
Point(204) = {0.3773660663507, 0.01164232375945, 0.0, basesize};
Point(205) = {0.3781053080569, 0.01165036975455, 0.0, basesize};
Point(206) = {0.378844549763, 0.01165726790172, 0.0, basesize};
Point(207) = {0.3795837914692, 0.01166303491916, 0.0, basesize};
Point(208) = {0.3803230331754, 0.01166768752507, 0.0, basesize};
Point(209) = {0.3810622748815, 0.01167124243764, 0.0, basesize};
Point(210) = {0.3818015165877, 0.01167371637508, 0.0, basesize};
Point(211) = {0.3825407582938, 0.01167512605558, 0.0, basesize};
Point(212) = {0.38328, 0.01167548819733, 0.0, basesize};
Point(213) = {0.385, 0.01167548819733, 0.0, basesize};
//Bottom Wall Back
Point(1001) = {0.21, -0.0270645, 0.0, basesize};
Point(1002) = {0.2280392417062, -0.0270645, 0.0, basesize};
Point(1003) = {0.2287784834123, -0.0270645, 0.0, basesize};
Point(1004) = {0.2295177251185, -0.0270645, 0.0, basesize};
Point(1005) = {0.2302569668246, -0.0270645, 0.0, basesize};
Point(1006) = {0.2309962085308, -0.0270645, 0.0, basesize};
Point(1007) = {0.231735450237, -0.0270645, 0.0, basesize};
Point(1008) = {0.2324746919431, -0.0270645, 0.0, basesize};
Point(1009) = {0.2332139336493, -0.0270645, 0.0, basesize};
Point(1010) = {0.2339531753555, -0.0270645, 0.0, basesize};
Point(1011) = {0.2346924170616, -0.02679430246686, 0.0, basesize};
Point(1012) = {0.2354316587678, -0.0262852999159, 0.0, basesize};
Point(1013) = {0.2361709004739, -0.02577629736494, 0.0, basesize};
Point(1014) = {0.2369101421801, -0.02526729481398, 0.0, basesize};
Point(1015) = {0.2376493838863, -0.02475829226302, 0.0, basesize};
Point(1016) = {0.2383886255924, -0.02424928971206, 0.0, basesize};
Point(1017) = {0.2391278672986, -0.0237402871611, 0.0, basesize};
Point(1018) = {0.2398671090047, -0.02323128461014, 0.0, basesize};
Point(1019) = {0.2406063507109, -0.02272228205918, 0.0, basesize};
Point(1020) = {0.2413455924171, -0.02221327950822, 0.0, basesize};
Point(1021) = {0.2420848341232, -0.02170427695726, 0.0, basesize};
Point(1022) = {0.2428240758294, -0.0211952744063, 0.0, basesize};
Point(1023) = {0.2435633175355, -0.02068627185534, 0.0, basesize};
Point(1024) = {0.2443025592417, -0.02017726930438, 0.0, basesize};
Point(1025) = {0.2450418009479, -0.01966826675342, 0.0, basesize};
Point(1026) = {0.245781042654, -0.01915926420245, 0.0, basesize};
Point(1027) = {0.2465202843602, -0.01865026165149, 0.0, basesize};
Point(1028) = {0.2472595260664, -0.01814125910053, 0.0, basesize};
Point(1029) = {0.2479987677725, -0.01763225654957, 0.0, basesize};
Point(1030) = {0.2487380094787, -0.01712325399861, 0.0, basesize};
Point(1031) = {0.2494772511848, -0.01661425144765, 0.0, basesize};
Point(1032) = {0.250216492891, -0.01610524889669, 0.0, basesize};
Point(1033) = {0.2509557345972, -0.01559624634573, 0.0, basesize};
Point(1034) = {0.2516949763033, -0.01508724379477, 0.0, basesize};
Point(1035) = {0.2524342180095, -0.01457824124381, 0.0, basesize};
Point(1036) = {0.2531734597156, -0.01406923869285, 0.0, basesize};
Point(1037) = {0.2539127014218, -0.01356023614189, 0.0, basesize};
Point(1038) = {0.254651943128, -0.01305123359093, 0.0, basesize};
Point(1039) = {0.2553911848341, -0.01254223104006, 0.0, basesize};
Point(1040) = {0.2561304265403, -0.01203324300793, 0.0, basesize};
Point(1041) = {0.2568696682464, -0.01153323105766, 0.0, basesize};
Point(1042) = {0.2576089099526, -0.01107308263704, 0.0, basesize};
Point(1043) = {0.2583481516588, -0.01065403753526, 0.0, basesize};
Point(1044) = {0.2590873933649, -0.0102747284778, 0.0, basesize};
Point(1045) = {0.2598266350711, -0.009933787576145, 0.0, basesize};
Point(1046) = {0.2605658767773, -0.009629846941815, 0.0, basesize};
Point(1047) = {0.2613051184834, -0.009361538686311, 0.0, basesize};
Point(1048) = {0.2620443601896, -0.009127494921139, 0.0, basesize};
Point(1049) = {0.2627836018957, -0.008926347757803, 0.0, basesize};
Point(1050) = {0.2635228436019, -0.008756729307808, 0.0, basesize};
Point(1051) = {0.2642620853081, -0.008617271682661, 0.0, basesize};
Point(1052) = {0.2650013270142, -0.008506606993866, 0.0, basesize};
Point(1053) = {0.2657405687204, -0.008423367352927, 0.0, basesize};
Point(1054) = {0.2664798104265, -0.008366184871351, 0.0, basesize};
Point(1055) = {0.2672190521327, -0.008333691646977, 0.0, basesize};
Point(1056) = {0.2679582938389, -0.008324504031647, 0.0, basesize};
Point(1057) = {0.268697535545, -0.008324500001239, 0.0, basesize};
Point(1058) = {0.2694367772512, -0.0083245, 0.0, basesize};
Point(1059) = {0.2701760189573, -0.0083245, 0.0, basesize};
Point(1060) = {0.2709152606635, -0.0083245, 0.0, basesize};
Point(1061) = {0.2716545023697, -0.0083245, 0.0, basesize};
Point(1062) = {0.2723937440758, -0.0083245, 0.0, basesize};
Point(1063) = {0.273132985782, -0.0083245, 0.0, basesize};
Point(1064) = {0.2738722274882, -0.0083245, 0.0, basesize};
Point(1065) = {0.2746114691943, -0.0083245, 0.0, basesize};
Point(1066) = {0.2753507109005, -0.0083245, 0.0, basesize};
Point(1067) = {0.2760899526066, -0.0083245, 0.0, basesize};
Point(1068) = {0.2768291943128, -0.0083245, 0.0, basesize};
Point(1069) = {0.277568436019, -0.0083245, 0.0, basesize};
Point(1070) = {0.2783076777251, -0.0083245, 0.0, basesize};
Point(1071) = {0.2790469194313, -0.0083245, 0.0, basesize};
Point(1072) = {0.2797861611374, -0.0083245, 0.0, basesize};
Point(1073) = {0.2805254028436, -0.0083245, 0.0, basesize};
Point(1074) = {0.2812646445498, -0.0083245, 0.0, basesize};
Point(1075) = {0.2820038862559, -0.0083245, 0.0, basesize};
Point(1076) = {0.2827431279621, -0.0083245, 0.0, basesize};
Point(1077) = {0.2834823696682, -0.0083245, 0.0, basesize};
Point(1078) = {0.2842216113744, -0.0083245, 0.0, basesize};
Point(1079) = {0.2849608530806, -0.0083245, 0.0, basesize};
Point(1080) = {0.2857000947867, -0.0083245, 0.0, basesize};
Point(1081) = {0.2864393364929, -0.0083245, 0.0, basesize};
Point(1082) = {0.2871785781991, -0.0083245, 0.0, basesize};
Point(1083) = {0.2879178199052, -0.0083245, 0.0, basesize};
Point(1084) = {0.2886570616114, -0.0083245, 0.0, basesize};
Point(1085) = {0.2893963033175, -0.0083245, 0.0, basesize};
Point(1086) = {0.2901355450237, -0.0083245, 0.0, basesize};
Point(1087) = {0.2908747867299, -0.0083245, 0.0, basesize};
Point(1088) = {0.291614028436, -0.0083245, 0.0, basesize};
Point(1089) = {0.2923532701422, -0.0083245, 0.0, basesize};
Point(1090) = {0.2930925118483, -0.0083245, 0.0, basesize};
Point(1091) = {0.2938317535545, -0.0083245, 0.0, basesize};
Point(1092) = {0.2945709952607, -0.0083245, 0.0, basesize};
Point(1093) = {0.2953102369668, -0.0083245, 0.0, basesize};
Point(1094) = {0.296049478673, -0.0083245, 0.0, basesize};
Point(1095) = {0.2967887203791, -0.0083245, 0.0, basesize};
Point(1096) = {0.2975279620853, -0.0083245, 0.0, basesize};
Point(1097) = {0.2982672037915, -0.0083245, 0.0, basesize};
Point(1098) = {0.2990064454976, -0.0083245, 0.0, basesize};
Point(1099) = {0.2997456872038, -0.0083245, 0.0, basesize};
Point(1100) = {0.30048492891, -0.0083245, 0.0, basesize};
Point(1101) = {0.3012241706161, -0.0083245, 0.0, basesize};
Point(1102) = {0.3019634123223, -0.0083245, 0.0, basesize};
Point(1103) = {0.3027026540284, -0.0083245, 0.0, basesize};
Point(1104) = {0.3034418957346, -0.0083245, 0.0, basesize};
Point(1105) = {0.3041811374408, -0.0083245, 0.0, basesize};
Point(1106) = {0.3049203791469, -0.0083245, 0.0, basesize};
Point(1107) = {0.3056596208531, -0.0083245, 0.0, basesize};
Point(1108) = {0.3063988625592, -0.0083245, 0.0, basesize};
Point(1109) = {0.3071381042654, -0.0083245, 0.0, basesize};
Point(1110) = {0.3078773459716, -0.0083245, 0.0, basesize};
Point(1111) = {0.3086165876777, -0.0083245, 0.0, basesize};
Point(1112) = {0.3093558293839, -0.0083245, 0.0, basesize};
Point(1113) = {0.31009507109, -0.0083245, 0.0, basesize};
Point(1114) = {0.3108343127962, -0.0083245, 0.0, basesize};
Point(1115) = {0.3115735545024, -0.0083245, 0.0, basesize};
Point(1116) = {0.3123127962085, -0.0083245, 0.0, basesize};
Point(1117) = {0.3130520379147, -0.0083245, 0.0, basesize};
Point(1118) = {0.3137912796209, -0.0083245, 0.0, basesize};
Point(1119) = {0.314530521327, -0.0083245, 0.0, basesize};
Point(1120) = {0.3152697630332, -0.0083245, 0.0, basesize};
Point(1121) = {0.3160090047393, -0.0083245, 0.0, basesize};
Point(1122) = {0.3167482464455, -0.0083245, 0.0, basesize};
Point(1123) = {0.3174874881517, -0.0083245, 0.0, basesize};
Point(1124) = {0.3182267298578, -0.0083245, 0.0, basesize};
Point(1125) = {0.318965971564, -0.0083245, 0.0, basesize};
Point(1126) = {0.3197052132701, -0.0083245, 0.0, basesize};
Point(1127) = {0.3204444549763, -0.0083245, 0.0, basesize};
Point(1128) = {0.3211836966825, -0.0083245, 0.0, basesize};
Point(1129) = {0.3219229383886, -0.0083245, 0.0, basesize};
Point(1130) = {0.3226621800948, -0.0083245, 0.0, basesize};
Point(1131) = {0.3234014218009, -0.0083245, 0.0, basesize};
Point(1132) = {0.3241406635071, -0.0083245, 0.0, basesize};
Point(1133) = {0.3248799052133, -0.0083245, 0.0, basesize};
Point(1134) = {0.3256191469194, -0.0083245, 0.0, basesize};
Point(1135) = {0.3263583886256, -0.0083245, 0.0, basesize};
Point(1136) = {0.3270976303318, -0.0083245, 0.0, basesize};
Point(1137) = {0.3278368720379, -0.0083245, 0.0, basesize};
Point(1138) = {0.3285761137441, -0.0083245, 0.0, basesize};
Point(1139) = {0.3293153554502, -0.0083245, 0.0, basesize};
Point(1140) = {0.3300545971564, -0.0083245, 0.0, basesize};
Point(1141) = {0.3307938388626, -0.0083245, 0.0, basesize};
Point(1142) = {0.3315330805687, -0.0083245, 0.0, basesize};
Point(1143) = {0.3322723222749, -0.0083245, 0.0, basesize};
Point(1144) = {0.333011563981, -0.0083245, 0.0, basesize};
Point(1145) = {0.3337508056872, -0.0083245, 0.0, basesize};
Point(1146) = {0.3344900473934, -0.0083245, 0.0, basesize};
Point(1147) = {0.3352292890995, -0.0083245, 0.0, basesize};
Point(1148) = {0.3359685308057, -0.0083245, 0.0, basesize};
Point(1149) = {0.3367077725118, -0.0083245, 0.0, basesize};
Point(1150) = {0.337447014218, -0.0083245, 0.0, basesize};
Point(1151) = {0.3381862559242, -0.0083245, 0.0, basesize};
Point(1152) = {0.3389254976303, -0.0083245, 0.0, basesize};
Point(1153) = {0.3396647393365, -0.0083245, 0.0, basesize};
Point(1154) = {0.3404039810427, -0.0083245, 0.0, basesize};
Point(1155) = {0.3411432227488, -0.0083245, 0.0, basesize};
Point(1156) = {0.341882464455, -0.0083245, 0.0, basesize};
Point(1157) = {0.3426217061611, -0.0083245, 0.0, basesize};
Point(1158) = {0.3433609478673, -0.0083245, 0.0, basesize};
Point(1159) = {0.3441001895735, -0.0083245, 0.0, basesize};
Point(1160) = {0.3448394312796, -0.0083245, 0.0, basesize};
Point(1161) = {0.3455786729858, -0.0083245, 0.0, basesize};
Point(1162) = {0.3463179146919, -0.0083245, 0.0, basesize};
Point(1163) = {0.3470571563981, -0.0083245, 0.0, basesize};
Point(1164) = {0.3477963981043, -0.0083245, 0.0, basesize};
Point(1165) = {0.3485356398104, -0.0083245, 0.0, basesize};
Point(1166) = {0.3492748815166, -0.0083245, 0.0, basesize};
Point(1167) = {0.3500141232227, -0.0083245, 0.0, basesize};
Point(1168) = {0.3507533649289, -0.0083245, 0.0, basesize};
Point(1169) = {0.3514926066351, -0.0083245, 0.0, basesize};
Point(1170) = {0.3522318483412, -0.0083245, 0.0, basesize};
Point(1171) = {0.3529710900474, -0.0083245, 0.0, basesize};
Point(1172) = {0.3537103317536, -0.0083245, 0.0, basesize};
Point(1173) = {0.3544495734597, -0.0083245, 0.0, basesize};
Point(1174) = {0.3551888151659, -0.0083245, 0.0, basesize};
Point(1175) = {0.355928056872, -0.0083245, 0.0, basesize};
Point(1176) = {0.3566672985782, -0.0083245, 0.0, basesize};
Point(1177) = {0.3574065402844, -0.0083245, 0.0, basesize};
Point(1178) = {0.3581457819905, -0.0083245, 0.0, basesize};
Point(1179) = {0.3588850236967, -0.0083245, 0.0, basesize};
Point(1180) = {0.3596242654028, -0.0083245, 0.0, basesize};
Point(1181) = {0.360363507109, -0.0083245, 0.0, basesize};
Point(1182) = {0.3611027488152, -0.0083245, 0.0, basesize};
Point(1183) = {0.3618419905213, -0.0083245, 0.0, basesize};
Point(1184) = {0.3625812322275, -0.0083245, 0.0, basesize};
Point(1185) = {0.3633204739336, -0.0083245, 0.0, basesize};
Point(1186) = {0.3640597156398, -0.0083245, 0.0, basesize};
Point(1187) = {0.364798957346, -0.0083245, 0.0, basesize};
Point(1188) = {0.3655381990521, -0.0083245, 0.0, basesize};
Point(1189) = {0.3662774407583, -0.0083245, 0.0, basesize};
Point(1190) = {0.3670166824645, -0.0083245, 0.0, basesize};
Point(1191) = {0.3677559241706, -0.0083245, 0.0, basesize};
Point(1192) = {0.3684951658768, -0.0083245, 0.0, basesize};
Point(1193) = {0.3692344075829, -0.0083245, 0.0, basesize};
Point(1194) = {0.3699736492891, -0.0083245, 0.0, basesize};
Point(1195) = {0.3707128909953, -0.0083245, 0.0, basesize};
Point(1196) = {0.3714521327014, -0.0083245, 0.0, basesize};
Point(1197) = {0.3721913744076, -0.0083245, 0.0, basesize};
Point(1198) = {0.3729306161137, -0.0083245, 0.0, basesize};
Point(1199) = {0.3736698578199, -0.0083245, 0.0, basesize};
Point(1200) = {0.3744090995261, -0.0083245, 0.0, basesize};
Point(1201) = {0.3751483412322, -0.0083245, 0.0, basesize};
Point(1202) = {0.3758875829384, -0.0083245, 0.0, basesize};
Point(1203) = {0.3766268246445, -0.0083245, 0.0, basesize};
Point(1204) = {0.3773660663507, -0.0083245, 0.0, basesize};
Point(1205) = {0.3781053080569, -0.0083245, 0.0, basesize};
Point(1206) = {0.378844549763, -0.0083245, 0.0, basesize};
Point(1207) = {0.3795837914692, -0.0083245, 0.0, basesize};
Point(1208) = {0.3803230331754, -0.0083245, 0.0, basesize};
Point(1209) = {0.3810622748815, -0.0083245, 0.0, basesize};
Point(1210) = {0.3818015165877, -0.0083245, 0.0, basesize};
Point(1211) = {0.38328, -0.0083245, 0.0, basesize};
Point(1212) = {0.384, -0.0083245, 0.0, basesize};
Point(1213) = {0.385, -0.0083245, 0.0, basesize};

//Make Lines
Spline(1000) = {1:213};     //Top back
Spline(1001) = {1001:1213}; //Bottom back

// Make the back surface
// Inlet
Line(400) = {1,1001};  //goes counter-clockwise

//Cavity Start
Point(450) = {0.65163,-0.0083245,0.0,basesize};

//Bottom of cavity
Point(451) = {0.65163,-0.0283245,0.0,basesize};
Point(452) = {0.70163,-0.0283245,0.0,basesize};
Point(453) = {0.72163,-0.0083245,0.0,basesize};
Point(454) = {0.72163+0.02,-0.0083245,0.0,basesize};

//Extend downstream a bit
Point(455) = {0.65163+0.335,-0.008324-(0.265-0.02)*Sin(2*Pi/180),0.0,basesize};
Point(456) = {0.65163+0.335,0.01167548819733,0.0,basesize};

//Make Cavity lines
Line(450) = {1213,450};
Line(451) = {450,451};
Line(452) = {451,452};
Line(453) = {452,453};
Line(454) = {453,454};
Line(455) = {454,455};

// Outlet
//Line(401) = {213,1213};  //goes counter-clockwise
Line(401) = {455,456};  //goes counter-clockwise

//Top wall
Line(457) = {456,213};  // goes counter-clockwise

//Create lineloop of this geometry
// start on the bottom left and go around clockwise
//Curve Loop(1) = { 400, 1001, -401, -1000 }; Plane Surface(1) = {1}; // the back wall
Curve Loop(1) = { 
    -400, // inlet (2)
    1000, // top nozzle (3)
    -457, // top extension to end (4)
    -401, // outlet (5)
    -455, // bottom expansion (6)
    -454, // post-cavity flat (7)
    -453, // cavity rear (slant) (8)
    -452, // cavity bottom (9)
    -451, // cavity front (10)
    -450, // isolator to cavity (11)
    -1001 // bottom nozzle (12)
}; 

Surface(1) = {1}; // the back wall

// surfaceVector contains in the following order:
// [0]	- front surface (opposed to source surface)
// [1] - extruded volume
// [n+1] - surfaces (belonging to nth line in "Curve Loop (1)") */
surface_vector[] = Extrude {0, 0, 0.035} { Surface{1}; };

//surface_vector[0], // front surface (opposing extruded surface)
//surface_vector[2], // isolator bottom
//surface_vector[3], // front of cavity?
//surface_vector[4], // bottom of cavity
//surface_vector[5], // cavity slant wall
//surface_vector[6], // post-cavity flat
//surface_vector[7], // expansion bottom
//surface_vector[8], // outlet
//surface_vector[9], // isolator top
//surface_vector[10], // nozzle top
//surface_vector[11], // inlet
//surface_vector[12], // nozzle bottom
//1                   // back surface (original)
//
//bottom right cavity corner {0.70163,-0.0283245,0.0}
//Cylinder { x0, y0, z0, xn, yn, zn, r }
Cylinder(100) = {0.70163, -0.0283245 + inj_h + inj_t/2., 0.035/2., inj_d, 0.0, 0.0, inj_t/2.0 };
injector_surface_vector[] = Boundary{Volume{100};};
// form union with isolator volume
union[] = BooleanUnion { Volume{surface_vector[1]}; Delete; }{Volume{100}; Delete; };
// Abs removes the directionality of the surface, so we can use in mesh generation (spacing)
surface_vector_full[] = Abs(Boundary{Volume{union[0]};});
//
//Printf("union length = %g", #union[]);
//Printf("surface length = %g", #surface_vector[]);
//For i In {0:#surface_vector[]-1}
    //Printf("surface_vector: %g",surface_vector[i]);
//EndFor
//Printf("surface length = %g", #injector_surface_vector[]);
//For i In {0:#injector_surface_vector[]-1}
    //Printf("injector_surface_vector: %g",injector_surface_vector[i]);
//EndFor
//For i In {0:#surface_vector_full[]-1}
    //Printf("surface_vector_full: %g",surface_vector_full[i]);
//EndFor

//surface_vector_full[1], // nozzle top
//surface_vector_full[2], // nozzle bottom
//surface_vector_full[3], // isolator back (aft)
//surface_vector_full[4], // isolator front (fore)
//surface_vector_full[5], // isolator top
//surface_vector_full[6], // isolator bottom
//surface_vector_full[7], // cavity front
//surface_vector_full[8], // cavity bottom
//surface_vector_full[9], // cavity slant
//surface_vector_full[10], // post cavity flat
//surface_vector_full[11], // expansion bottom
//surface_vector_full[12], // outlet
//surface_vector_full[13], // injector wall
//surface_vector_full[14], // injector inlet

//Physical Volume("fluid_domain") = surface_vector[1];
Physical Volume("fluid_domain") = union[0];
Physical Surface("inflow") = 1; // inlet
Physical Surface("outflow") = surface_vector_full[12]; // outlet
Physical Surface("injection") = surface_vector_full[14]; // injection
Physical Surface("flow") = {
    1,
    surface_vector_full[12],
    surface_vector_full[14]
};
Physical Surface('wall') = {
surface_vector_full[1], // nozzle top
surface_vector_full[2], // nozzle bottom
surface_vector_full[3], // isolator back (aft)
surface_vector_full[4], // isolator front (fore)
surface_vector_full[5], // isolator top
surface_vector_full[6], // isolator bottom
surface_vector_full[7], // cavity front
surface_vector_full[8], // cavity bottom
surface_vector_full[9], // cavity slant
surface_vector_full[10], // post cavity flat
surface_vector_full[11], // expansion bottom
surface_vector_full[13] // injector wall
};

// Create distance field from surfaces for wall meshing, excludes cavity, injector
Field[1] = Distance;
Field[1].SurfacesList = {
surface_vector_full[1], // nozzle top
surface_vector_full[2], // nozzle bottom
surface_vector_full[3], // isolator back (aft)
surface_vector_full[4], // isolator front (fore)
surface_vector_full[5], // isolator top
surface_vector_full[6], // isolator bottom
surface_vector_full[10], // post cavity flat
surface_vector_full[11] // expansion bottom
};
Field[1].Sampling = 1000;
//
//Create threshold field that varrries element size near boundaries
Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = nozzlesize / boundratio;
Field[2].SizeMax = isosize;
Field[2].DistMin = 0.00002;
Field[2].DistMax = 0.005;
Field[2].StopAtDistMax = 1;

// Create distance field from curves, cavity only
Field[11] = Distance;
Field[11].SurfacesList = {
surface_vector_full[7], // cavity front
surface_vector_full[8], // cavity bottom
surface_vector_full[9] // cavity slant
};
Field[11].Sampling = 1000;

//Create threshold field that varrries element size near boundaries
Field[12] = Threshold;
Field[12].InField = 11;
Field[12].SizeMin = cavitysize / boundratiocavity;
Field[12].SizeMax = cavitysize;
Field[12].DistMin = 0.00002;
Field[12].DistMax = 0.005;
Field[12].StopAtDistMax = 1;

// Create distance field from curves, injector only
Field[13] = Distance;
Field[13].SurfacesList = {
surface_vector_full[13] // injector wall
};
Field[13].Sampling = 1000;

//Create threshold field that varrries element size near boundaries
Field[14] = Threshold;
Field[14].InField = 13;
Field[14].SizeMin = injectorsize / boundratioinjector;
Field[14].SizeMax = injectorsize;
Field[14].DistMin = 0.000001;
Field[14].DistMax = 0.0005;
Field[14].StopAtDistMax = 1;

nozzle_start = 0.27;
nozzle_end = 0.30;
//  background mesh size in the isolator (downstream of the nozzle)
Field[3] = Box;
Field[3].XMin = nozzle_end;
Field[3].XMax = 1.0;
Field[3].YMin = -1.0;
Field[3].YMax = 1.0;
Field[3].ZMin = -1.0;
Field[3].ZMax = 1.0;
Field[3].VIn = isosize;
Field[3].VOut = bigsize;

// background mesh size upstream of the inlet
Field[4] = Box;
Field[4].XMin = 0.;
Field[4].XMax = nozzle_start;
Field[4].YMin = -1.0;
Field[4].YMax = 1.0;
Field[4].ZMin = -1.0;
Field[4].ZMax = 1.0;
Field[4].VIn = inletsize;
Field[4].VOut = bigsize;

// background mesh size in the nozzle throat
Field[5] = Box;
Field[5].XMin = nozzle_start;
Field[5].XMax = nozzle_end;
Field[5].YMin = -1.0;
Field[5].YMax = 1.0;
Field[5].ZMin = -1.0;
Field[5].ZMax = 1.0;
Field[5].Thickness = 0.10;    // interpolate from VIn to Vout over a distance around the box
Field[5].VIn = nozzlesize;
Field[5].VOut = bigsize;

// background mesh size in the cavity region
cavity_start = 0.65;
cavity_end = 0.73;
Field[6] = Box;
Field[6].XMin = cavity_start;
Field[6].XMax = cavity_end;
Field[6].YMin = -1.0;
//Field[6].YMax = -0.003;
Field[6].YMax = 0.0;
Field[6].ZMin = -1.0;
Field[6].ZMax = 1.0;
Field[6].Thickness = 0.10;    // interpolate from VIn to Vout over a distance around the box
Field[6].VIn = cavitysize;
Field[6].VOut = bigsize;

// background mesh size in the injection region
injector_start_x = 0.69;
injector_end_x = 0.75;
//injector_start_y = -0.0225;
injector_start_y = -0.021;
injector_end_y = -0.026;
injector_start_z = 0.0175 - 0.002;
injector_end_z = 0.0175 + 0.002;
Field[7] = Box;
Field[7].XMin = injector_start_x;
Field[7].XMax = injector_end_x;
Field[7].YMin = injector_start_y;
Field[7].YMax = injector_end_y;
Field[7].ZMin = injector_start_z;
Field[7].ZMax = injector_end_z;
Field[7].Thickness = 0.10;    // interpolate from VIn to Vout over a distance around the cylinder
//Field[7] = Cylinder;
//Field[7].XAxis = 1;
//Field[7].YCenter = -0.0225295;
//Field[7].ZCenter = 0.0157;
//Field[7].Radius = 0.003;
Field[7].VIn = injectorsize;
Field[7].VOut = bigsize;

// background mesh size in the shear region
shear_start_x = 0.65;
shear_end_x = 0.73;
shear_start_y = -0.004;
shear_end_y = -0.01;
shear_start_z = -1.0;
shear_end_z = 1.0;
Field[8] = Box;
Field[8].XMin = shear_start_x;
Field[8].XMax = shear_end_x;
Field[8].YMin = shear_start_y;
Field[8].YMax = shear_end_y;
Field[8].ZMin = shear_start_z;
Field[8].ZMax = shear_end_z;
Field[8].Thickness = 0.10;  
Field[8].VIn = shearsize;
Field[8].VOut = bigsize;

// take the minimum of all defined meshing fields
Field[100] = Min;
//Field[100].FieldsList = {2, 3, 4, 5, 6, 7, 12, 14};
Field[100].FieldsList = {2, 3, 4, 5, 6, 7, 8, 12, 14};
//Field[100].FieldsList = {2};
Background Field = 100;

Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
