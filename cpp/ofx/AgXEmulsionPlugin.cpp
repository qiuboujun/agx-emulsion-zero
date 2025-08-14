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
        std::cerr << "  dst depth: " << dst->getPixelDepth() << " (expected: " << kOfxBitDepthFloat << ")" << std::endl;
        std::cerr << "  dst components: " << dst->getPixelComponents() << " (expected: " << kOfxImageComponentRGBA << ")" << std::endl;
        std::cerr << "  src depth: " << src->getPixelDepth() << " (expected: " << kOfxBitDepthFloat << ")" << std::endl;
        
        assert(dst->getPixelDepth() == kOfxBitDepthFloat && dst->getPixelComponents() == kOfxImageComponentRGBA);
        assert(src->getPixelDepth() == kOfxBitDepthFloat);
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
        params.io.input_color_space = "sRGB";
        params.io.input_cctf_decoding = false;
        params.io.output_color_space = "sRGB";
        params.io.output_cctf_encoding = false;
        params.io.full_image = true;
        
        // Re-enable LUTs after fixing CUDA launch issues
        bool enable_luts = true; // Re-enabled after fixing grid dimension calculation
        params.settings.use_camera_lut = enable_luts;
        params.settings.use_enlarger_lut = enable_luts;
        params.settings.use_scanner_lut = enable_luts;
        params.settings.apply_masking_couplers = true;
        std::cerr << "  base params set (LUTs " << (enable_luts ? "enabled" : "DISABLED for debugging") << ")" << std::endl;

        // Fetch OFX params
        std::cerr << "AgXEmulsionProcessor: Fetching OFX parameters..." << std::endl;
        IntParam* lutRes = fetchIntParam("lutResolution");
        BooleanParam* enablePrint = fetchBooleanParam("enablePrint");
        ChoiceParam* film = fetchChoiceParam("filmStock");
        ChoiceParam* paper = fetchChoiceParam("printPaper");
        BooleanParam* paperGlare = fetchBooleanParam("paperGlare");
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

        int lutR = 32; if (lutRes) lutRes->getValue(lutR); params.settings.lut_resolution = lutR;
        bool doPrint = true; if (enablePrint) enablePrint->getValue(doPrint);
        int filmIdx = 1; if (film) film->getValue(filmIdx);
        int paperIdx = 0; if (paper) paper->getValue(paperIdx);
        bool glareOn = false; if (paperGlare) paperGlare->getValue(glareOn);
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

        // Run the real process (prepare copies for async safety)
        std::cerr << "AgXEmulsionProcessor: Preparing async process run..." << std::endl;
        auto paramsCopy = params;                  // copy Params
        auto inputCopy = input.copy();             // deep copy input buffer
        std::cerr << "AgXEmulsionProcessor: Params and input copied for async" << std::endl;
        
        nc::NdArray<float> out; // Declare outside try block
        
        // Add timeout mechanism (thread uses owned copies; no dangling refs on timeout)
        std::cerr << "AgXEmulsionProcessor: Starting process with timeout (async, by-value)..." << std::endl;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
        const int timeout_seconds = 45; // relaxed timeout to reduce spurious aborts
        
        try {
            // Run process in a separate thread with timeout. Capture by value to avoid dangling refs.
            std::future<nc::NdArray<float>> future = std::async(
                std::launch::async,
                [paramsCopy = std::move(paramsCopy), inputCopy = std::move(inputCopy)]() mutable {
                    agx::process::Process procLocal(paramsCopy);
                    return procLocal.run(inputCopy);
                }
            );
            
            // Wait for completion or timeout
            if (future.wait_for(std::chrono::seconds(timeout_seconds)) == std::future_status::ready) {
                out = future.get();
                std::cerr << "  process completed successfully" << std::endl;
                std::cerr << "  output shape: " << out.shape().rows << "x" << out.shape().cols << std::endl;
                
                // Check output range
                float outMin = 1e6f, outMax = -1e6f;
                for (int h = 0; h < height; ++h) {
                    for (int w = 0; w < width; ++w) {
                        outMin = std::min(outMin, std::min(std::min(out(h, w*3+0), out(h, w*3+1)), out(h, w*3+2)));
                        outMax = std::max(outMax, std::max(std::max(out(h, w*3+0), out(h, w*3+1)), out(h, w*3+2)));
                    }
                }
                std::cerr << "  output range: [" << outMin << ", " << outMax << "]" << std::endl;
                
            } else {
                std::cerr << "ERROR: Process::run() timed out after " << timeout_seconds << " seconds (async task continues safely)" << std::endl;
                // Drop the future without calling get; async task owns its data copies.
                return;
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
        // Scatter back RGB to RGBA buffer, keep original A
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                size_t pi = (static_cast<size_t>(h) * width + static_cast<size_t>(w)) * 4;
                outRGBA[pi + 0] = out(h, w*3 + 0);
                outRGBA[pi + 1] = out(h, w*3 + 1);
                outRGBA[pi + 2] = out(h, w*3 + 2);
                outRGBA[pi + 3] = inRGBA[pi + 3];
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
        desc.setHostFrameThreading(false);
        desc.setSupportsMultiResolution(false);
        desc.setSupportsTiles(true);
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
        srcClip->setSupportsTiles(true);
        srcClip->setIsMask(false);

        ClipDescriptor *dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
        dstClip->addSupportedComponent(ePixelComponentRGBA);
        dstClip->setSupportsTiles(true);

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
            lutRes->setDefault(32);
            lutRes->setRange(8, 64);
        }
        {
            auto *glare = desc.defineBooleanParam("paperGlare");
            glare->setLabel("Paper Glare");
            glare->setDefault(false);
        }
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


