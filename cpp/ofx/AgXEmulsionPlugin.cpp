#include <cassert>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <iostream>
#include <future> // Required for std::future and std::async
#include <chrono> // Required for std::chrono

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"

#include "NumCpp.hpp"
#include "process.hpp"
#include "config.hpp"

// Optional CUDA demo kept for reference
extern "C" void RunCudaOfxBridge(int width, int height, float exposureEV,
                                 const float* hostInRGBA, float* hostOutRGBA);

using namespace OFX;

class AgXEmulsionProcessor : public ImageEffect {
public:
    AgXEmulsionProcessor(OfxImageEffectHandle handle)
        : ImageEffect(handle) {
        std::cerr << "AgXEmulsionProcessor: Constructor called with handle " << handle << std::endl;
        agx::config::initialize_config();
        std::cerr << "AgXEmulsionProcessor: Config initialized" << std::endl;
    }

    void render(const RenderArguments &args) override {
        std::cerr << "AgXEmulsionProcessor: render() called at time " << args.time << std::endl;
        std::cerr << "AgXEmulsionProcessor: Fetching images..." << std::endl;
        
        std::unique_ptr<Image> dst(dstClip_->fetchImage(args.time));
        std::unique_ptr<const Image> src(srcClip_->fetchImage(args.time));
        
        if (!src) {
            std::cerr << "ERROR: Failed to fetch source image!" << std::endl;
            return;
        }
        if (!dst) {
            std::cerr << "ERROR: Failed to fetch destination image!" << std::endl;
            return;
        }
        std::cerr << "AgXEmulsionProcessor: Images fetched successfully" << std::endl;

        OfxRectI rect = args.renderWindow;
        const int width = rect.x2 - rect.x1;
        const int height = rect.y2 - rect.y1;
        std::cerr << "AgXEmulsionProcessor: Render window " << width << "x" << height << std::endl;
        std::cerr << "AgXEmulsionProcessor: Rect bounds: x1=" << rect.x1 << ", y1=" << rect.y1 << ", x2=" << rect.x2 << ", y2=" << rect.y2 << std::endl;

        // Expect float RGBA from host
        std::cerr << "AgXEmulsionProcessor: Checking pixel format..." << std::endl;
        std::cerr << "  dst depth: " << dst->getPixelDepth() << " (expected: " << OFX::eBitDepthFloat << ")" << std::endl;
        std::cerr << "  dst components: " << dst->getPixelComponents() << " (expected: " << OFX::ePixelComponentRGBA << ")" << std::endl;
        std::cerr << "  src depth: " << src->getPixelDepth() << " (expected: " << OFX::eBitDepthFloat << ")" << std::endl;
        
        assert(dst->getPixelDepth() == OFX::eBitDepthFloat && dst->getPixelComponents() == OFX::ePixelComponentRGBA);
        assert(src->getPixelDepth() == OFX::eBitDepthFloat);
        std::cerr << "AgXEmulsionProcessor: Pixel format validation passed" << std::endl;

        // Gather into contiguous RGBA buffer
        std::cerr << "AgXEmulsionProcessor: Allocating buffers..." << std::endl;
        const size_t numPixels = static_cast<size_t>(width) * static_cast<size_t>(height);
        std::cerr << "  numPixels: " << numPixels << std::endl;
        std::cerr << "  buffer size: " << (numPixels * 4 * sizeof(float)) << " bytes" << std::endl;
        
        std::vector<float> inRGBA(numPixels * 4);
        std::vector<float> outRGBA(numPixels * 4);
        
        PixelComponentEnum srcComp = src->getPixelComponents();
        bool srcHasAlpha = (srcComp == ePixelComponentRGBA);
        std::cerr << "  src components: " << srcComp << ", hasAlpha: " << (srcHasAlpha ? "true" : "false") << std::endl;
        
        std::cerr << "AgXEmulsionProcessor: Gathering input data..." << std::endl;
        int validPixels = 0;
        int nullPixels = 0;
        float minVal = 1e6f, maxVal = -1e6f;
        
        for (int y = rect.y1; y < rect.y2; ++y) {
            for (int x = rect.x1; x < rect.x2; ++x) {
                int w = x - rect.x1; int h = y - rect.y1;
                size_t idx = (static_cast<size_t>(h) * width + static_cast<size_t>(w)) * 4;
                const float* sp = reinterpret_cast<const float*>(src->getPixelAddress(x, y));
                if (!sp) { 
                    inRGBA[idx+0]=inRGBA[idx+1]=inRGBA[idx+2]=0.f; 
                    inRGBA[idx+3]=1.f; 
                    nullPixels++;
                    continue; 
                }
                validPixels++;
                inRGBA[idx+0] = sp[0];
                inRGBA[idx+1] = sp[1];
                inRGBA[idx+2] = sp[2];
                inRGBA[idx+3] = srcHasAlpha ? sp[3] : 1.0f;
                
                minVal = std::min(minVal, std::min(std::min(sp[0], sp[1]), sp[2]));
                maxVal = std::max(maxVal, std::max(std::max(sp[0], sp[1]), sp[2]));
            }
        }
        std::cerr << "  gathered pixels: " << validPixels << " valid, " << nullPixels << " null" << std::endl;
        std::cerr << "  input range: [" << minVal << ", " << maxVal << "]" << std::endl;

        // Build Params from OFX controls
        std::cerr << "AgXEmulsionProcessor: Building process parameters..." << std::endl;
        agx::process::Params params;
        // Default; may be overridden by UI (inputColorSpace)
        params.io.input_color_space = "ACES2065-1";
        params.io.input_cctf_decoding = false; // keep linear by default (parity)
        // Default output color space to ACES2065-1 linear
        params.io.output_color_space = "ACES2065-1";
        params.io.output_cctf_encoding = false; // linear
        params.io.full_image = true;
        
        // LUT toggles controlled by UI (defaults set below in describeInContext)
        params.settings.use_camera_lut = false;
        params.settings.use_enlarger_lut = false;
        params.settings.use_scanner_lut = false;
        params.settings.apply_masking_couplers = true;
        std::cerr << "  base params set (LUTs controlled by UI)" << std::endl;

        // Fetch OFX params
        std::cerr << "AgXEmulsionProcessor: Fetching OFX parameters..." << std::endl;
        IntParam* lutRes = fetchIntParam("lutResolution");
        BooleanParam* enablePrint = fetchBooleanParam("enablePrint");
        ChoiceParam* film = fetchChoiceParam("filmStock");
        ChoiceParam* paper = fetchChoiceParam("printPaper");
        BooleanParam* paperGlare = fetchBooleanParam("paperGlare");
        // New toggles and input color space
        BooleanParam* halationToggle = fetchBooleanParam("halation");
        BooleanParam* grainToggle = fetchBooleanParam("grain");
        ChoiceParam* inputCS = fetchChoiceParam("inputColorSpace");
        // LUT toggles
        BooleanParam* cameraLUT = fetchBooleanParam("useCameraLUT");
        BooleanParam* enlargerLUT = fetchBooleanParam("useEnlargerLUT");
        BooleanParam* scannerLUT = fetchBooleanParam("useScannerLUT");
        ChoiceParam* outputCS = fetchChoiceParam("outputColorSpace");
        // DIR couplers UI
        BooleanParam* enableCouplers = fetchBooleanParam("enableCouplers");
        DoubleParam* dirAmount = fetchDoubleParam("dirAmount");
        DoubleParam* dirRatioR = fetchDoubleParam("dirRatioR");
        DoubleParam* dirRatioG = fetchDoubleParam("dirRatioG");
        DoubleParam* dirRatioB = fetchDoubleParam("dirRatioB");
        DoubleParam* dirDiffSize = fetchDoubleParam("dirDiffusionUm");
        DoubleParam* dirDiffInter = fetchDoubleParam("dirDiffusionInterlayer");
        DoubleParam* dirHighShift = fetchDoubleParam("dirHighExposureShift");
        // remove deprecated disable* params if present in older bundles
        DoubleParam* exposureEV = fetchDoubleParam("exposureEV");
        BooleanParam* autoExposure = fetchBooleanParam("autoExposure");
        DoubleParam* printExposure = fetchDoubleParam("printExposure");
        BooleanParam* printExposureComp = fetchBooleanParam("printExposureCompensation");
        IntParam* yFilterShift = fetchIntParam("yFilterShift");
        IntParam* mFilterShift = fetchIntParam("mFilterShift");
        
        std::cerr << "  parameter pointers: lutRes=" << (lutRes ? "valid" : "null") 
                  << ", enablePrint=" << (enablePrint ? "valid" : "null")
                  << ", film=" << (film ? "valid" : "null")
                  << ", paper=" << (paper ? "valid" : "null") << std::endl;

        int lutR = 17; if (lutRes) lutRes->getValue(lutR); params.settings.lut_resolution = lutR;  // Reduced from 32 to 17 (17³ = 4,913 vs 32³ = 32,768)
        bool doPrint = true; if (enablePrint) enablePrint->getValue(doPrint);
        int filmIdx = 1; if (film) film->getValue(filmIdx);
        int paperIdx = 0; if (paper) paper->getValue(paperIdx);
        bool glareOn = false; if (paperGlare) paperGlare->getValue(glareOn);
        // Map toggles: unchecked disables
        bool halOn = false; if (halationToggle) halationToggle->getValue(halOn); params.settings.disable_halation = !halOn;
        bool grainOn = false; if (grainToggle) grainToggle->getValue(grainOn); params.settings.disable_grain = !grainOn;
        // LUTs
        bool camL = false; if (cameraLUT) cameraLUT->getValue(camL); params.settings.use_camera_lut = camL;
        bool enlL = true; if (enlargerLUT) enlargerLUT->getValue(enlL); params.settings.use_enlarger_lut = enlL;
        bool scnL = true; if (scannerLUT) scannerLUT->getValue(scnL); params.settings.use_scanner_lut = scnL;
        int csIdx = 0; if (inputCS) inputCS->getValue(csIdx);
        if (csIdx == 0) {
            params.io.input_color_space = "ACES2065-1";
            params.io.input_cctf_decoding = false; // linear
        } else {
            params.io.input_color_space = "sRGB";
            params.io.input_cctf_decoding = false; // parity with Python runs
        }
        // DIR couplers
        params.settings.apply_dir_couplers = true; if (enableCouplers) { bool ec=true; enableCouplers->getValue(ec); params.settings.apply_dir_couplers = ec; }
        if (dirAmount) { double v; dirAmount->getValue(v); params.settings.dir_amount = v; }
        if (dirRatioR) { double v; dirRatioR->getValue(v); params.settings.dir_ratio_rgb[0] = v; }
        if (dirRatioG) { double v; dirRatioG->getValue(v); params.settings.dir_ratio_rgb[1] = v; }
        if (dirRatioB) { double v; dirRatioB->getValue(v); params.settings.dir_ratio_rgb[2] = v; }
        if (dirDiffSize) { double v; dirDiffSize->getValue(v); params.settings.dir_diffusion_size_um = v; }
        if (dirDiffInter) { double v; dirDiffInter->getValue(v); params.settings.dir_diffusion_interlayer = v; }
        if (dirHighShift) { double v; dirHighShift->getValue(v); params.settings.dir_high_exposure_shift = v; }
        params.settings.disable_halation = !halOn;
        params.settings.disable_grain = !grainOn;
        double ev = 0.0; if (exposureEV) exposureEV->getValue(ev); params.camera.exposure_compensation_ev = (float)ev;
        bool ae = true; if (autoExposure) autoExposure->getValue(ae); params.camera.auto_exposure = ae;
        double pexp = 1.0; if (printExposure) printExposure->getValue(pexp); params.enlarger.print_exposure = (float)pexp;
        bool pec = true; if (printExposureComp) printExposureComp->getValue(pec); params.enlarger.print_exposure_compensation = pec;
        int yfs = 0; if (yFilterShift) yFilterShift->getValue(yfs); params.enlarger.y_filter_shift = (float)yfs;
        int mfs = 0; if (mFilterShift) mFilterShift->getValue(mfs); params.enlarger.m_filter_shift = (float)mfs;
        
        std::cerr << "  parameter values: lutR=" << lutR << ", doPrint=" << doPrint 
                  << ", filmIdx=" << filmIdx << ", paperIdx=" << paperIdx 
                  << ", glareOn=" << glareOn << ", ev=" << ev << ", ae=" << ae 
                  << ", pexp=" << pexp << ", pec=" << pec << ", yfs=" << yfs << ", mfs=" << mfs << std::endl;
        int ocsIdx = 0; if (outputCS) outputCS->getValue(ocsIdx);
        params.io.output_color_space = (ocsIdx==0?"ACES2065-1":"sRGB");
        params.io.output_cctf_encoding = (ocsIdx==1); // encode gamma only for sRGB
        std::cerr << "  toggles: halation=" << (halOn?1:0) << ", grain=" << (grainOn?1:0)
                  << ", inputCS=" << (csIdx==0?"ACES2065-1":"sRGB")
                  << ", outputCS=" << (ocsIdx==0?"ACES2065-1":"sRGB")
                  << ", cameraLUT=" << (camL?1:0) << ", enlargerLUT=" << (enlL?1:0) << ", scannerLUT=" << (scnL?1:0)
                  << std::endl;
        std::cerr << "  toggles: halation=" << (halOn?1:0)
                  << ", grain=" << (grainOn?1:0) << std::endl;

        static const char* filmOptions[] = {
            "kodak_portra_160_auc","kodak_portra_400_auc","kodak_portra_800_auc",
            "kodak_ektar_100_auc","kodak_gold_200_auc","kodak_ultramax_400_auc",
            "kodak_vision3_50d_uc","kodak_vision3_250d_uc","kodak_vision3_200t_uc",
            "kodak_vision3_500t_uc","fujifilm_c200_auc","fujifilm_xtra_400_auc",
            "fujifilm_pro_400h_auc"
        };
        static const char* paperOptions[] = {
            "kodak_portra_endura_uc","kodak_endura_premier_uc","kodak_ektacolor_edge_uc",
            "kodak_supra_endura_uc","fujifilm_crystal_archive_typeii_uc","kodak_2383_uc","kodak_2393_uc"
        };
        if (filmIdx >= 0 && filmIdx < (int)(sizeof(filmOptions)/sizeof(filmOptions[0]))) {
            params.profiles.negative = filmOptions[filmIdx];
            std::cerr << "  selected film: " << params.profiles.negative << std::endl;
        } else {
            std::cerr << "  WARNING: invalid film index " << filmIdx << std::endl;
        }
        if (doPrint) {
            params.io.compute_negative = false;
            if (paperIdx >= 0 && paperIdx < (int)(sizeof(paperOptions)/sizeof(paperOptions[0]))) {
                params.profiles.print_paper = paperOptions[paperIdx];
                std::cerr << "  selected paper: " << params.profiles.print_paper << std::endl;
            } else {
                std::cerr << "  WARNING: invalid paper index " << paperIdx << std::endl;
            }
            params.settings.apply_paper_glare = glareOn;
        } else {
            params.io.compute_negative = true; // stop at negative stage
            std::cerr << "  stopping at negative stage" << std::endl;
        }

        // Build H x (W*3) input from inRGBA
        std::cerr << "AgXEmulsionProcessor: Building input array..." << std::endl;
        nc::NdArray<float> input(height, width * 3);
        std::cerr << "  input array shape: " << input.shape().rows << "x" << input.shape().cols << std::endl;
        
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                size_t idx = (static_cast<size_t>(h) * width + static_cast<size_t>(w)) * 4;
                input(h, w*3 + 0) = inRGBA[idx + 0];
                input(h, w*3 + 1) = inRGBA[idx + 1];
                input(h, w*3 + 2) = inRGBA[idx + 2];
            }
        }
        std::cerr << "  input array populated" << std::endl;

        // Run the real process synchronously to avoid allocator/thread teardown issues
        std::cerr << "AgXEmulsionProcessor: Creating process instance..." << std::endl;
        agx::process::Process proc(params);
        std::cerr << "AgXEmulsionProcessor: Running process synchronously..." << std::endl;
        
        nc::NdArray<float> out; // Declare outside try block
        
        // Timing
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        
        try {
            // Run synchronously
            out = proc.run(input);
            std::cerr << "  process completed successfully" << std::endl;
            std::cerr << "  output shape: " << out.shape().rows << "x" << out.shape().cols << std::endl;
            
            // Check output range and sample pixels
            float outMin = 1e6f, outMax = -1e6f;
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    outMin = std::min(outMin, std::min(std::min(out(h, w*3+0), out(h, w*3+1)), out(h, w*3+2)));
                    outMax = std::max(outMax, std::max(std::max(out(h, w*3+0), out(h, w*3+1)), out(h, w*3+2)));
                }
            }
            std::cerr << "  output range: [" << outMin << ", " << outMax << "]" << std::endl;
            
            // Sample a few pixels for debugging
            if (height > 10 && width > 10) {
                std::cerr << "  sample pixels:" << std::endl;
                for (int i = 0; i < 3; ++i) {
                    int h = height / 4 + i * height / 8;
                    int w = width / 4 + i * width / 8;
                    std::cerr << "    [" << h << "," << w << "] = (" 
                              << out(h, w*3+0) << ", " << out(h, w*3+1) << ", " << out(h, w*3+2) << ")" << std::endl;
                }
            }
            
        } catch (const std::exception& e) {
            std::cerr << "ERROR: Process::run() threw exception: " << e.what() << std::endl;
            return;
        } catch (...) {
            std::cerr << "ERROR: Process::run() threw unknown exception" << std::endl;
            return;
        }
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start);
        std::cerr << "  process execution time: " << elapsed.count() << "ms" << std::endl;

        std::cerr << "AgXEmulsionProcessor: Scattering output data..." << std::endl;
        
        // Scatter back RGB to RGBA buffer; force opaque alpha to avoid host compositing issues
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                size_t pi = (static_cast<size_t>(h) * width + static_cast<size_t>(w)) * 4;
                outRGBA[pi + 0] = out(h, w*3 + 0);
                outRGBA[pi + 1] = out(h, w*3 + 1);
                outRGBA[pi + 2] = out(h, w*3 + 2);
                outRGBA[pi + 3] = 1.0f;
            }
        }
        std::cerr << "  output buffer populated" << std::endl;

        // Write back to Resolve
        std::cerr << "AgXEmulsionProcessor: Writing back to destination..." << std::endl;
        int writtenPixels = 0;
        int failedWrites = 0;
        for (int y = rect.y1; y < rect.y2; ++y) {
            for (int x = rect.x1; x < rect.x2; ++x) {
                int w = x - rect.x1; int h = y - rect.y1;
                size_t idx = (static_cast<size_t>(h) * width + static_cast<size_t>(w)) * 4;
                float* dp = reinterpret_cast<float*>(dst->getPixelAddress(x, y));
                if (!dp) { 
                    failedWrites++;
                    continue; 
                }
                writtenPixels++;
                dp[0] = outRGBA[idx+0];
                dp[1] = outRGBA[idx+1];
                dp[2] = outRGBA[idx+2];
                dp[3] = outRGBA[idx+3];
            }
        }
        std::cerr << "  writeback completed: " << writtenPixels << " pixels written, " << failedWrites << " failed" << std::endl;
        
        // Release images per OFX API (delete the Image wrappers)
        if (src) { delete src.release(); }
        if (dst) { delete dst.release(); }
        
        std::cerr << "AgXEmulsionProcessor: render() completed successfully" << std::endl;
    }

    void changedParam(const InstanceChangedArgs &/*args*/, const std::string &/*paramName*/) override {}
    void changedClip(const InstanceChangedArgs &/*args*/, const std::string &/*clipName*/) override {}

    void setSrcClip(Clip* clip) { 
        std::cerr << "AgXEmulsionProcessor: setSrcClip called with " << (clip ? "valid" : "null") << " clip" << std::endl;
        srcClip_ = clip; 
    }
    void setDstClip(Clip* clip) { 
        std::cerr << "AgXEmulsionProcessor: setDstClip called with " << (clip ? "valid" : "null") << " clip" << std::endl;
        dstClip_ = clip; 
    }

private:
    Clip* srcClip_ = nullptr;
    Clip* dstClip_ = nullptr;
};

class AgXEmulsionPluginFactory : public PluginFactoryHelper<AgXEmulsionPluginFactory> {
public:
    AgXEmulsionPluginFactory() : PluginFactoryHelper<AgXEmulsionPluginFactory>("com.jq.agxemulsion", 1, 0) {}
    ~AgXEmulsionPluginFactory() {
        std::cerr << "AgXEmulsionPluginFactory: Destructor called" << std::endl;
    }

    void describe(OFX::ImageEffectDescriptor &desc) override {
        std::cerr << "AgXEmulsionPluginFactory: describe() called" << std::endl;
        desc.setLabel("AgX Emulsion");
        desc.setPluginGrouping("OpenFX JQ");
        desc.addSupportedContext(eContextFilter);
        desc.addSupportedContext(eContextGeneral);
        desc.addSupportedBitDepth(eBitDepthFloat);
        desc.setSingleInstance(false);
        desc.setHostFrameThreading(true);
        desc.setSupportsMultiResolution(false);
        desc.setSupportsTiles(false);
        desc.setTemporalClipAccess(false);
        desc.setRenderTwiceAlways(false);
        desc.setSupportsMultipleClipPARs(false);
        std::cerr << "AgXEmulsionPluginFactory: describe() completed" << std::endl;
    }

    void describeInContext(OFX::ImageEffectDescriptor &desc, ContextEnum /*context*/) override {
        std::cerr << "AgXEmulsionPluginFactory: describeInContext() called" << std::endl;
        ClipDescriptor *srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
        srcClip->addSupportedComponent(ePixelComponentRGBA);
        srcClip->setTemporalClipAccess(false);
        srcClip->setSupportsTiles(false);
        srcClip->setIsMask(false);

        ClipDescriptor *dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
        dstClip->addSupportedComponent(ePixelComponentRGBA);
        dstClip->setSupportsTiles(false);

        // Parameters
        {
            auto *film = desc.defineChoiceParam("filmStock");
            film->setLabel("Film Stock");
            film->appendOption("kodak_portra_160_auc");
            film->appendOption("kodak_portra_400_auc");
            film->appendOption("kodak_portra_800_auc");
            film->appendOption("kodak_ektar_100_auc");
            film->appendOption("kodak_gold_200_auc");
            film->appendOption("kodak_ultramax_400_auc");
            film->appendOption("kodak_vision3_50d_uc");
            film->appendOption("kodak_vision3_250d_uc");
            film->appendOption("kodak_vision3_200t_uc");
            film->appendOption("kodak_vision3_500t_uc");
            film->appendOption("fujifilm_c200_auc");
            film->appendOption("fujifilm_xtra_400_auc");
            film->appendOption("fujifilm_pro_400h_auc");
            film->setDefault(1);
        }
        {
            auto *paper = desc.defineChoiceParam("printPaper");
            paper->setLabel("Print Paper");
            paper->appendOption("kodak_portra_endura_uc");
            paper->appendOption("kodak_endura_premier_uc");
            paper->appendOption("kodak_ektacolor_edge_uc");
            paper->appendOption("kodak_supra_endura_uc");
            paper->appendOption("fujifilm_crystal_archive_typeii_uc");
            paper->appendOption("kodak_2383_uc");
            paper->appendOption("kodak_2393_uc");
            paper->setDefault(0);
        }
        {
            auto *usePaper = desc.defineBooleanParam("enablePrint");
            usePaper->setLabel("Enable Print Stage");
            usePaper->setDefault(true);
        }
        {
            auto *lutRes = desc.defineIntParam("lutResolution");
            lutRes->setLabel("LUT Resolution");
            lutRes->setDefault(17);  // Reduced from 32 to 17 (17³ = 4,913 vs 32³ = 32,768)
            lutRes->setRange(8, 32); // Reduced max from 64 to 32
        }
        {
            auto *p = desc.defineBooleanParam("useCameraLUT");
            p->setLabel("Use Camera LUT");
            p->setDefault(false);
        }
        {
            auto *p = desc.defineBooleanParam("useEnlargerLUT");
            p->setLabel("Use Enlarger LUT");
            p->setDefault(true);
        }
        {
            auto *p = desc.defineBooleanParam("useScannerLUT");
            p->setLabel("Use Scanner LUT");
            p->setDefault(true);
        }
        {
            auto *glare = desc.defineBooleanParam("paperGlare");
            glare->setLabel("Paper Glare");
            glare->setDefault(false);
        }
        {
            auto *hal = desc.defineBooleanParam("halation");
            hal->setLabel("Halation");
            hal->setDefault(false); // unchecked = disabled by default
        }
        {
            auto *gr = desc.defineBooleanParam("grain");
            gr->setLabel("Grain");
            gr->setDefault(false); // unchecked = disabled by default
        }
        {
            auto *ics = desc.defineChoiceParam("inputColorSpace");
            ics->setLabel("Input Color Space");
            ics->appendOption("ACES2065-1 (linear)");
            ics->appendOption("sRGB");
            ics->setDefault(0); // default to ACES2065-1 linear
        }
        {
            auto *ocs = desc.defineChoiceParam("outputColorSpace");
            ocs->setLabel("Output Color Space");
            ocs->appendOption("ACES2065-1 (linear)");
            ocs->appendOption("sRGB");
            ocs->setDefault(0); // default ACES2065-1 linear
        }
        {
            auto *ec = desc.defineBooleanParam("enableCouplers");
            ec->setLabel("Enable Couplers");
            ec->setDefault(true);
        }
        {
            auto *p = desc.defineDoubleParam("dirAmount");
            p->setLabel("DIR Amount");
            p->setDefault(1.0);
            p->setDisplayRange(0.0, 3.0);
        }
        {
            auto *p = desc.defineDoubleParam("dirRatioR"); p->setLabel("DIR Ratio R"); p->setDefault(1.0); p->setDisplayRange(0.0, 3.0);
        }
        {
            auto *p = desc.defineDoubleParam("dirRatioG"); p->setLabel("DIR Ratio G"); p->setDefault(1.0); p->setDisplayRange(0.0, 3.0);
        }
        {
            auto *p = desc.defineDoubleParam("dirRatioB"); p->setLabel("DIR Ratio B"); p->setDefault(1.0); p->setDisplayRange(0.0, 3.0);
        }
        {
            auto *p = desc.defineDoubleParam("dirDiffusionUm"); p->setLabel("DIR Diffusion (um)"); p->setDefault(10.0); p->setDisplayRange(0.0, 100.0);
        }
        {
            auto *p = desc.defineDoubleParam("dirDiffusionInterlayer"); p->setLabel("DIR Diffusion Interlayer"); p->setDefault(2.0); p->setDisplayRange(0.0, 10.0);
        }
        {
            auto *p = desc.defineDoubleParam("dirHighExposureShift"); p->setLabel("DIR High Exposure Shift"); p->setDefault(0.0); p->setDisplayRange(0.0, 5.0);
        }
        // removed deprecated disableHalation/disableGrain params
        {
            auto *expEV = desc.defineDoubleParam("exposureEV");
            expEV->setLabel("Exposure EV");
            expEV->setDefault(0.0);
            expEV->setDisplayRange(-5.0, 5.0);
        }
        {
            auto *autoExp = desc.defineBooleanParam("autoExposure");
            autoExp->setLabel("Auto Exposure");
            autoExp->setDefault(true);
        }
        {
            auto *printExposure = desc.defineDoubleParam("printExposure");
            printExposure->setLabel("Print Exposure (x)");
            printExposure->setDefault(1.0);
            printExposure->setDisplayRange(0.0, 10.0);
        }
        {
            auto *expComp = desc.defineBooleanParam("printExposureCompensation");
            expComp->setLabel("Use Print Exposure Compensation");
            expComp->setDefault(true);
        }
        {
            auto *yShift = desc.defineIntParam("yFilterShift");
            yShift->setLabel("Y Filter Shift");
            yShift->setDefault(0);
            yShift->setRange(-170, 170);
        }
        {
            auto *mShift = desc.defineIntParam("mFilterShift");
            mShift->setLabel("M Filter Shift");
            mShift->setDefault(0);
            mShift->setRange(-170, 170);
        }
        std::cerr << "AgXEmulsionPluginFactory: describeInContext() completed" << std::endl;
    }

    ImageEffect* createInstance(OfxImageEffectHandle handle, ContextEnum /*context*/) override {
        std::cerr << "AgXEmulsionPluginFactory: createInstance() called with handle " << handle << std::endl;
        auto *inst = new AgXEmulsionProcessor(handle);
        std::cerr << "AgXEmulsionProcessor instance created" << std::endl;
        inst->setSrcClip(inst->fetchClip(kOfxImageEffectSimpleSourceClipName));
        std::cerr << "AgXEmulsionProcessor src clip set" << std::endl;
        inst->setDstClip(inst->fetchClip(kOfxImageEffectOutputClipName));
        std::cerr << "AgXEmulsionProcessor dst clip set" << std::endl;
        std::cerr << "AgXEmulsionPluginFactory: createInstance() completed" << std::endl;
        return inst;
    }
};

// Register factory instance with host
namespace OFX { namespace Plugin {
void getPluginIDs(OFX::PluginFactoryArray &id) {
    std::cerr << "getPluginIDs() called - registering AgXEmulsionPlugin" << std::endl;
    static AgXEmulsionPluginFactory p;
    id.push_back(&p);
    std::cerr << "getPluginIDs() completed - plugin registered" << std::endl;
}
} } // namespace OFX::Plugin


