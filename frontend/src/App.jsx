import React, { useState, useRef, useEffect } from 'react';
import { 
  AlertCircle, CheckCircle, Upload, Link, Type, Loader2, 
  AlertTriangle, FileText, ShieldCheck, Sparkles, Github, 
  Cpu, Database, Code, Zap, Brain, Layout, Server, GitBranch, ArrowRight, ExternalLink, Globe, Lightbulb, HelpCircle
} from 'lucide-react';

// !!! FINAL DEPLOYMENT FIX: HARDCODE LIVE URL (Safest Method for Immediate Functionality) !!!
// Using the direct Render URL avoids 'process is not defined' errors in client-side code.
const API_BASE_URL = "https://factcheck-backend-xiyn.onrender.com";

export default function App() {
  const [currentPage, setCurrentPage] = useState('home'); // 'home' or 'about'
  const [activeTab, setActiveTab] = useState('text'); // 'text', 'url', 'image'
  const [inputText, setInputText] = useState('');
  const [inputUrl, setInputUrl] = useState('');
  const [selectedFile, setSelectedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  
  // New state to track if Tailwind styles are ready
  const [isStyleReady, setIsStyleReady] = useState(false);
  
  // Refs for scrolling
  const howItWorksRef = useRef(null);

  // Set Document Title on Mount
  useEffect(() => {
    document.title = "FactCheck AI - Misinformation Detector";
  }, []);

  // --- ROBUST FIX: Polling for Tailwind ---
  useEffect(() => {
    if (window.tailwind) {
      setIsStyleReady(true);
      return;
    }

    const scriptId = 'tailwind-cdn-script';
    if (!document.getElementById(scriptId)) {
      const script = document.createElement('script');
      script.id = scriptId;
      script.src = "https://cdn.tailwindcss.com";
      script.async = true;
      document.head.appendChild(script);
    }

    const intervalId = setInterval(() => {
      if (window.tailwind) {
        setIsStyleReady(true);
        clearInterval(intervalId);
      }
    }, 50);

    const timeoutId = setTimeout(() => {
      clearInterval(intervalId);
      setIsStyleReady(true); 
    }, 3000);

    return () => {
      clearInterval(intervalId);
      clearTimeout(timeoutId);
    };
  }, []);

  // Navigation Helper
  const navigateTo = (page) => {
    setCurrentPage(page);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleNavToSection = (ref) => {
    if (currentPage !== 'home') {
        setCurrentPage('home');
        setTimeout(() => {
            if(ref.current) ref.current.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    } else {
        if(ref.current) ref.current.scrollIntoView({ behavior: 'smooth' });
    }
  };

  const handleTabChange = (tab) => {
    setActiveTab(tab);
    setResult(null);
    setError(null);
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      
      // Client-side size validation (1MB = 1048576 bytes)
      if (file.size > 1024 * 1024) {
        setError("File size exceeds 1MB limit. Please upload a smaller image (or crop it).");
        setSelectedFile(null);
        e.target.value = null; // Reset the input
        return;
      }
      
      setSelectedFile(file);
      setError(null);
    }
  };

  const analyzeText = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    try {
      // Use API_BASE_URL + /analyze path for robust connection
      const response = await fetch(`${API_BASE_URL}/analyze`, { 
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      });
      if (!response.ok) throw new Error('Failed to analyze text');
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Could not connect to the backend. The service may be temporarily down or a network configuration error exists.");
    } finally {
      setLoading(false);
    }
  };

  const analyzeUrl = async () => {
    if (!inputUrl.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/predict_url`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: inputUrl }),
      });
      if (!response.ok) throw new Error('Failed to scrape URL');
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("Could not analyze URL. The site might block scrapers or the backend is down.");
    } finally {
      setLoading(false);
    }
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;
    setLoading(true);
    setError(null);
    
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(`${API_BASE_URL}/predict_image`, {
        method: 'POST',
        body: formData,
      });
      
      if (response.status === 413) {
        throw new Error('File is too large for the server (Max 1MB).');
      }
      
      if (!response.ok) {
         // Try to get error message from backend
         const errData = await response.json().catch(() => ({}));
         throw new Error(errData.detail || 'Failed to process image');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Error processing image. Ensure Tesseract is installed on the server.");
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (activeTab === 'text') analyzeText();
    if (activeTab === 'url') analyzeUrl();
    if (activeTab === 'image') analyzeImage();
  };

  // Helper to determine styles based on classification
  const getResultStyles = (classification) => {
    if (classification === 'Fake') {
        return {
            container: 'bg-rose-50 border-rose-100 text-rose-700',
            shadow: 'shadow-rose-500/10',
            bar: 'bg-rose-600',
            icon: <AlertTriangle className="w-6 h-6" />,
            label: "Likely Misinformation"
        };
    } else if (classification === 'Real') {
        return {
            container: 'bg-emerald-50 border-emerald-100 text-emerald-700',
            shadow: 'shadow-emerald-500/10',
            bar: 'bg-emerald-500',
            icon: <ShieldCheck className="w-6 h-6" />,
            label: "Credible Source"
        };
    } else { // Unverified
        return {
            container: 'bg-amber-50 border-amber-100 text-amber-700',
            shadow: 'shadow-amber-500/10',
            bar: 'bg-amber-500',
            icon: <HelpCircle className="w-6 h-6" />,
            label: "Unverified / Needs Sources"
        };
    }
  };

  if (!isStyleReady) {
    return (
      <div style={{
        height: '100vh',
        width: '100vw',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        fontFamily: 'system-ui, -apple-system, sans-serif',
        backgroundColor: '#f9fafb',
        color: '#4b5563'
      }}>
        <div style={{ marginBottom: '1rem' }}>
           <Loader2 style={{ width: '48px', height: '48px', animation: 'spin 1s linear infinite' }} />
        </div>
        <h2 style={{ fontSize: '1.25rem', fontWeight: '600', marginBottom: '0.5rem' }}>Initializing System...</h2>
        <p style={{ fontSize: '0.875rem', opacity: 0.8 }}>Loading interface and configurations</p>
      </div>
    );
  }

  // --- SUB-COMPONENT: ABOUT PAGE ---
  const AboutPage = () => (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-700 pb-24">
        <div className="max-w-3xl mx-auto">
            <button 
                onClick={() => navigateTo('home')} 
                className="mb-8 flex items-center gap-2 text-indigo-600 font-semibold hover:text-indigo-800 transition-colors group"
            >
                <ArrowRight className="w-4 h-4 rotate-180 group-hover:-translate-x-1 transition-transform" /> 
                Back to Analyzer
            </button>
            <div className="bg-white/60 backdrop-blur-xl rounded-3xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-white/60 ring-1 ring-gray-100/50 p-8 md:p-12">
                <h1 className="text-4xl md:text-5xl font-black text-gray-900 mb-8 tracking-tight">
                    Why <span className="text-indigo-700">FactCheck AI</span>?
                </h1>
                <div className="prose prose-lg text-gray-600 leading-relaxed">
                    <p className="text-xl font-medium text-gray-800 mb-8">
                        In an age where information travels faster than truth, misinformation has become one of the most significant challenges of our digital society.
                    </p>
                    <p className="mb-6">
                        We built <strong>FactCheck AI</strong> as a response to this growing problem. What started as a collective curiosity about how Natural Language Processing (NLP) could be applied to real-world problems quickly evolved into this project. Our goal was simple: provide a tool that acts as a first line of defense against sensationalism.
                    </p>
                    <div className="my-10 p-6 bg-indigo-50/50 rounded-2xl border border-indigo-100">
                        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center gap-2">
                            <Lightbulb className="w-5 h-5 text-amber-500" />
                            The Motivation
                        </h3>
                        <p className="text-base text-gray-700">
                            During scrolling sessions on social media, we noticed how difficult it was to verify claims without opening ten different tabs. We wanted to create something that could ingest a headline, a link, or even a screenshot, and give an immediate "probabilistic check" on its credibility.
                        </p>
                    </div>
                    <h3 className="text-2xl font-bold text-gray-900 mb-6">How It Was Built</h3>
                    <p className="mb-6">
                        This project combines several modern technologies to create a seamless experience. It leverages <strong>DistilBERT</strong>, a smaller, faster, cheaper and lighter version of BERT, to understand the semantic context of news titles and articles.
                    </p>
                    <div className="grid md:grid-cols-2 gap-4 not-prose mb-8">
                        <div className="bg-white p-4 rounded-xl border border-gray-100 shadow-sm flex items-start gap-3">
                            <div className="p-2 bg-blue-50 text-blue-600 rounded-lg shrink-0"><Brain className="w-5 h-5"/></div>
                            <div>
                                <strong className="block text-gray-900 text-sm mb-1">AI Model</strong>
                                <span className="text-xs text-gray-500">Fine-tuned on 50k+ news articles to detect sensationalism and bias.</span>
                            </div>
                        </div>
                        <div className="bg-white p-4 rounded-xl border border-gray-100 shadow-sm flex items-start gap-3">
                            <div className="p-2 bg-emerald-50 text-emerald-600 rounded-lg shrink-0"><Server className="w-5 h-5"/></div>
                            <div>
                                <strong className="block text-gray-900 text-sm mb-1">Backend</strong>
                                <span className="text-xs text-gray-500">Python & FastAPI for high-performance async inference.</span>
                            </div>
                        </div>
                    </div>
                    <p>
                        This is an open-source project intended for educational purposes. It represents a step towards a future where AI helps us navigate the complexities of the information age with greater confidence and clarity.
                    </p>
                </div>
            </div>
        </div>
    </div>
  );

  const resultStyle = result ? getResultStyles(result.classification) : null;

  return (
    <div 
      className="min-h-screen bg-gradient-to-br from-gray-50 to-indigo-50 text-gray-900 font-sans relative overflow-x-hidden selection:bg-cyan-100 selection:text-indigo-900"
      style={{ fontFamily: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' }}
    >
      
      {/* Background Ambience */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
         <div className="absolute top-[-10%] left-[-10%] w-[50vw] h-[50vw] bg-indigo-200/20 rounded-full blur-[120px]" />
         <div className="absolute bottom-[-10%] right-[-10%] w-[50vw] h-[50vw] bg-cyan-100/40 rounded-full blur-[120px]" />
      </div>

      {/* Navbar */}
      <nav className="fixed top-0 w-full z-50 bg-white border-b border-gray-200 shadow-sm transition-all duration-300">
        <div className="max-w-5xl mx-auto px-6 h-20 flex items-center justify-between">
          
          {/* Logo - NOW CLICKABLE TO GO TO ABOUT PAGE */}
          <div 
            className="flex items-center gap-3 cursor-pointer group" 
            onClick={() => navigateTo('about')}
            title="Read about the project"
          >
            <div className="w-10 h-10 bg-indigo-700 rounded-lg flex items-center justify-center shadow-md shadow-indigo-200 group-hover:bg-indigo-800 transition-colors">
              <ShieldCheck className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900 tracking-tight group-hover:text-indigo-700 transition-colors">
              FactCheck <span className="text-cyan-500">AI</span>
            </span>
          </div>

          {/* Menu Items */}
          <div className="hidden md:flex items-center gap-8">
            <button 
                onClick={() => navigateTo('home')} 
                className={`text-sm font-semibold transition-colors ${currentPage === 'home' ? 'text-indigo-700' : 'text-gray-600 hover:text-indigo-700'}`}
            >
                Home
            </button>
            <button 
                onClick={() => handleNavToSection(howItWorksRef)} 
                className="text-sm font-semibold text-gray-600 hover:text-indigo-700 transition-colors"
            >
                How It Works
            </button>
            <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-gray-800 transition-colors">
              <Github className="w-5 h-5" />
            </a>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="max-w-3xl mx-auto px-6 pt-36 pb-20">
        
        {currentPage === 'about' ? <AboutPage /> : (
            <>
                {/* Minimalist Hero */}
                <div className="text-center mb-16 animate-in fade-in slide-in-from-bottom-2 duration-700 ease-out fill-mode-forwards">
                  <h1 className="text-5xl md:text-6xl font-black text-gray-900 mb-6 tracking-tight leading-tight">
                    Check the facts. <br className="hidden md:block"/> Instantly.
                  </h1>
                  <p className="text-gray-600 text-xl font-medium max-w-2xl mx-auto leading-relaxed mb-4">
                    AI-powered misinformation detection using Natural Language Processing and DistilBERT
                  </p>
                  <p className="text-gray-400 text-base max-w-lg mx-auto">
                    Upload text, links, or screenshots to verify credibility in seconds.
                  </p>
                </div>

                {/* Input Section */}
                <div className="bg-white/60 backdrop-blur-xl rounded-3xl shadow-[0_8px_30px_rgb(0,0,0,0.04)] border border-white/60 ring-1 ring-gray-100/50 relative overflow-hidden transition-all duration-300 animate-in fade-in slide-in-from-bottom-3 duration-1000 delay-150 fill-mode-forwards">
                    
                    <div className="text-center pt-8 pb-0">
                      <h2 className="text-2xl font-bold text-gray-900 tracking-tight">Analyze Information</h2>
                    </div>

                    <div className="mx-8 mt-6 relative flex border-b border-gray-200/70">
                      {['text', 'url', 'image'].map((tab) => (
                        <button
                          key={tab}
                          onClick={() => handleTabChange(tab)}
                          className={`flex-1 py-4 flex items-center justify-center gap-2 text-sm font-semibold transition-colors duration-[160ms] ease-out relative z-10 group
                            ${activeTab === tab 
                              ? 'text-indigo-700' 
                              : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50/30 rounded-t-lg'}
                          `}
                        >
                          <span className={`transition-opacity duration-200 ${activeTab === tab ? 'opacity-100' : 'opacity-60 group-hover:opacity-100'}`}>
                            {tab === 'text' && <Type className="w-4 h-4" />}
                            {tab === 'url' && <Link className="w-4 h-4" />}
                            {tab === 'image' && <Upload className="w-4 h-4" />}
                          </span>
                          <span className="capitalize">{tab}</span>
                        </button>
                      ))}
                      
                      <div 
                        className="absolute bottom-0 h-[2px] bg-indigo-600 transition-all duration-[180ms] ease-out z-20"
                        style={{
                            width: '33.333%',
                            left: activeTab === 'text' ? '0%' : activeTab === 'url' ? '33.333%' : '66.666%'
                        }}
                      />
                    </div>

                    <div className="p-8 pt-6">
                      <form onSubmit={handleSubmit} className="space-y-6">
                        <div key={activeTab} className="animate-in fade-in duration-150 ease-out">
                            {activeTab === 'text' && (
                                <textarea
                                className="w-full h-40 p-4 rounded-xl bg-gray-50/50 border border-gray-200 focus:bg-white focus:border-indigo-500/30 focus:ring-4 focus:ring-indigo-500/10 outline-none resize-none text-gray-700 placeholder:text-gray-400 transition-all text-base leading-relaxed shadow-inner"
                                placeholder="Paste the news article or statement here"
                                value={inputText}
                                onChange={(e) => setInputText(e.target.value)}
                                />
                            )}

                            {activeTab === 'url' && (
                                <input
                                type="url"
                                className="w-full p-4 rounded-xl bg-gray-50/50 border border-gray-200 focus:bg-white focus:border-indigo-500/30 focus:ring-4 focus:ring-indigo-500/10 outline-none text-gray-700 placeholder:text-gray-400 transition-all text-base shadow-inner"
                                placeholder="Enter the link to the news source"
                                value={inputUrl}
                                onChange={(e) => setInputUrl(e.target.value)}
                                />
                            )}

                            {activeTab === 'image' && (
                                <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:bg-white/80 hover:border-indigo-400 transition-all cursor-pointer relative group bg-gray-50/30">
                                <input 
                                    type="file" 
                                    accept="image/*"
                                    onChange={handleFileChange}
                                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
                                />
                                <div className="flex flex-col items-center gap-3">
                                    <div className="w-12 h-12 rounded-full bg-indigo-50 text-indigo-700 flex items-center justify-center group-hover:scale-110 transition-transform shadow-sm">
                                    <Upload className="w-6 h-6" />
                                    </div>
                                    <div className="text-sm text-gray-600 font-medium">
                                    {selectedFile ? selectedFile.name : "Upload a screenshot of the article"}
                                    </div>
                                    {/* Added Warning Here */}
                                    <div className="text-xs text-amber-500 font-bold bg-amber-50 px-3 py-1 rounded-full border border-amber-100 flex items-center gap-1">
                                        <AlertCircle className="w-3 h-3" />
                                        Max file size: 1MB
                                    </div>
                                </div>
                                </div>
                            )}
                        </div>

                        <button
                          type="submit"
                          disabled={loading || (activeTab === 'text' && !inputText) || (activeTab === 'url' && !inputUrl) || (activeTab === 'image' && !selectedFile)}
                          className="w-full bg-indigo-700 hover:bg-indigo-600 disabled:bg-gray-200 disabled:text-gray-400 text-white font-bold py-4 rounded-xl transition-all duration-200 flex flex-col items-center justify-center gap-1 shadow-md shadow-indigo-900/10 hover:shadow-lg hover:shadow-indigo-900/20 hover:-translate-y-px active:translate-y-px focus:ring-4 focus:ring-indigo-500/20 outline-none min-h-[64px]"
                        >
                          {loading ? (
                            <>
                              <div className="flex items-center gap-2">
                                 <Loader2 className="animate-spin w-5 h-5" />
                                 <span>Analyzing content using DistilBERT...</span>
                              </div>
                              <span className="text-xs font-normal opacity-80">Evaluating linguistic patterns and semantic cues</span>
                            </>
                          ) : (
                            "Analyze Credibility"
                          )}
                        </button>
                      </form>

                      {error && (
                        <div className="mt-6 p-4 bg-rose-50 border border-rose-100 text-rose-600 rounded-xl flex items-center gap-3 text-sm animate-in fade-in slide-in-from-top-2">
                          <AlertCircle className="w-5 h-5 shrink-0" />
                          {error}
                        </div>
                      )}
                    </div>
                </div>

                {/* Results Area */}
                {result && resultStyle && (
                  <div className="mt-12 animate-in fade-in slide-in-from-bottom-4 duration-700">
                    <div className={`p-1.5 rounded-3xl bg-white/55 backdrop-blur-md border border-white/60 shadow-xl ${resultStyle.shadow}`}>
                        <div className="bg-white rounded-[22px] p-8 md:p-10 border border-gray-100">
                            
                            <div className="text-center mb-8">
                               <h3 className="text-2xl font-bold text-gray-900 mb-2">Credibility Assessment</h3>
                               <div className="w-16 h-1 bg-gray-100 mx-auto rounded-full"></div>
                            </div>

                            <div className="flex flex-col items-center justify-center gap-4 mb-8">
                                 <div className={`flex items-center gap-3 px-6 py-3 rounded-full border-2 ${resultStyle.container}`}>
                                     {resultStyle.icon}
                                     <span className="text-lg font-bold tracking-tight">
                                        {resultStyle.label}
                                     </span>
                                 </div>
                                 
                                 <div className="text-gray-500 font-medium">
                                    Confidence Score: <span className="text-gray-900 font-bold">{(Math.max(result.confidenceReal, result.confidenceFake) * 100).toFixed(1)}%</span>
                                 </div>
                            </div>

                            <div className="w-full bg-gray-100 h-3 rounded-full overflow-hidden mb-10 max-w-md mx-auto relative shadow-inner">
                               <div 
                                 className={`h-full rounded-full transition-all duration-1000 ${resultStyle.bar}`}
                                 style={{ width: `${Math.max(result.confidenceReal, result.confidenceFake) * 100}%` }}
                               />
                            </div>

                            {/* EXPLANATION SECTION */}
                            {result.explanation && (
                                <div className="bg-indigo-50/50 rounded-2xl p-6 border border-indigo-100 mb-6 text-center">
                                    <h4 className="font-bold text-indigo-900 mb-2">Analysis Summary</h4>
                                    <p className="text-indigo-800 text-sm leading-relaxed font-medium">
                                        {result.explanation}
                                    </p>
                                </div>
                            )}

                            {/* SUGGESTION SECTION - NEW! */}
                            {result.suggestion && (
                                <div className="bg-amber-50/50 rounded-2xl p-6 border border-amber-100 mb-6 text-center">
                                    <h4 className="font-bold text-amber-900 mb-2 flex items-center justify-center gap-2">
                                        <Lightbulb className="w-4 h-4 text-amber-600" />
                                        How to Improve Credibility
                                    </h4>
                                    <p className="text-amber-800 text-sm leading-relaxed font-medium">
                                        {result.suggestion}
                                    </p>
                                </div>
                            )}

                            <div className="bg-gray-50 rounded-2xl p-6 border border-gray-200 mb-6">
                                <h4 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
                                    <Brain className="w-4 h-4 text-indigo-700"/>
                                    Why this result?
                                </h4>
                                <ul className="space-y-3 text-sm text-gray-600">
                                    {result.factors && result.factors.length > 0 ? (
                                        result.factors.map((factor, idx) => (
                                            <li key={idx} className="flex items-start gap-2">
                                                <span className="text-lg leading-none mt-0.5" role="img" aria-label="indicator">{factor.startsWith('🚩') ? '🚩' : factor.startsWith('✅') ? '✅' : 'ℹ️'}</span>
                                                <span className={factor.startsWith('🚩') ? 'text-rose-700' : factor.startsWith('✅') ? 'text-emerald-700' : 'text-gray-600'}>
                                                    {factor.replace(/^[🚩✅ℹ️]\s*/, '')}
                                                </span>
                                            </li>
                                        ))
                                    ) : (
                                        <li className="flex items-start gap-2">
                                            <span className="text-gray-400 mt-0.5">•</span>
                                            No specific linguistic triggers were found.
                                        </li>
                                    )}
                                </ul>
                            </div>
                            
                            <div className="grid md:grid-cols-2 gap-6 mb-6">
                                {result.related_news && result.related_news.length > 0 && (
                                    <div className="bg-white rounded-2xl p-6 border border-blue-100 shadow-sm shadow-blue-50/50">
                                        <h4 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
                                            <Globe className="w-4 h-4 text-blue-600"/>
                                            Live Related News
                                        </h4>
                                        <div className="space-y-3">
                                            {result.related_news.map((item, idx) => (
                                                <a 
                                                    key={idx}
                                                    href={item.url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="block p-3 rounded-xl bg-gray-50 hover:bg-blue-50 border border-gray-100 hover:border-blue-200 transition-all group"
                                                >
                                                    <div className="text-xs font-bold text-blue-600 mb-1">{item.source}</div>
                                                    <div className="text-sm font-semibold text-gray-800 group-hover:text-blue-700 line-clamp-2 leading-snug">
                                                        {item.title}
                                                    </div>
                                                </a>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {result.verification_tools && result.verification_tools.length > 0 && (
                                    <div className="bg-white rounded-2xl p-6 border border-indigo-100 shadow-sm shadow-indigo-50/50">
                                        <h4 className="font-bold text-gray-900 mb-4 flex items-center gap-2">
                                            <Link className="w-4 h-4 text-indigo-700"/>
                                            Verification Tools
                                        </h4>
                                        <div className="space-y-3">
                                            {result.verification_tools.map((source, idx) => (
                                                <a 
                                                    key={idx}
                                                    href={source.url}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="flex items-center justify-between p-3 rounded-xl bg-gray-50 hover:bg-indigo-50 border border-gray-100 hover:border-indigo-200 transition-all group"
                                                >
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-8 h-8 rounded-full bg-white border border-gray-200 flex items-center justify-center text-indigo-600 font-bold text-xs shrink-0">
                                                            {source.source[0]}
                                                        </div>
                                                        <span className="text-sm font-semibold text-gray-700 group-hover:text-indigo-800">
                                                            {source.source}
                                                        </span>
                                                    </div>
                                                    <ExternalLink className="w-4 h-4 text-gray-400 group-hover:text-indigo-500 transition-colors" />
                                                </a>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                            
                            <div className="text-center">
                                <p className="text-xs text-gray-400 max-w-sm mx-auto">
                                    This prediction is probabilistic and based on linguistic analysis. It should not replace human judgment.
                                </p>
                            </div>

                        </div>
                    </div>
                  </div>
                )}
                
                {/* How It Works Section */}
                <div className="mt-24 mb-20 scroll-mt-24" ref={howItWorksRef}>
                    <div className="text-center mb-12">
                        <h2 className="text-2xl font-bold text-gray-900">How the System Works</h2>
                    </div>
                    
                    <div className="grid md:grid-cols-4 gap-6 relative">
                        <div className="hidden md:block absolute top-8 left-0 w-full h-0.5 bg-gray-100 -z-10 transform scale-x-90"></div>
                        
                        {[
                            { icon: Upload, title: "Input", desc: "User provides text, URL, or image" },
                            { icon: Sparkles, title: "Clean", desc: "Content is preprocessed & normalized" },
                            { icon: Brain, title: "Analyze", desc: "DistilBERT analyzes semantic patterns" },
                            { icon: ShieldCheck, title: "Predict", desc: "Model returns credibility score" }
                        ].map((step, idx) => (
                            <div key={idx} className="bg-white p-6 rounded-2xl border border-gray-100 text-center shadow-[0_4px_20px_rgb(0,0,0,0.03)] hover:shadow-[0_8px_30px_rgb(0,0,0,0.06)] transition-all hover:-translate-y-1 duration-300">
                                <div className="w-16 h-16 mx-auto bg-gray-50 rounded-full flex items-center justify-center mb-4 text-cyan-500 border border-gray-100">
                                    <step.icon className="w-8 h-8" />
                                </div>
                                <h3 className="font-bold text-gray-900 mb-2">{step.title}</h3>
                                <p className="text-sm text-gray-500">{step.desc}</p>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Tech Stack Section */}
                <div className="mb-24">
                    <div className="text-center mb-12">
                        <h2 className="text-2xl font-bold text-gray-900">Technology Used</h2>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                        {[
                            { icon: Brain, label: "DistilBERT", url: "https://huggingface.co/docs/transformers/model_doc/distilbert" },
                            { icon: Code, label: "Python", url: "https://www.python.org/" },
                            { icon: Zap, label: "FastAPI", url: "https://fastapi.tiangolo.com/" },
                            { icon: Layout, label: "React", url: "https://react.dev/" },
                            { icon: Cpu, label: "PyTorch", url: "https://pytorch.org/" },
                        ].map((tech, idx) => (
                            <a 
                                key={idx} 
                                href={tech.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="bg-white p-4 rounded-xl border border-gray-100 flex flex-col items-center justify-center gap-2 hover:border-cyan-400 hover:shadow-md transition-all duration-200 cursor-pointer group"
                            >
                                <tech.icon className="w-6 h-6 text-indigo-400 group-hover:text-indigo-600 transition-colors" />
                                <span className="text-sm font-semibold text-gray-600 group-hover:text-gray-900 transition-colors">{tech.label}</span>
                            </a>
                        ))}
                    </div>
                </div>
            </>
        )}

      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-12">
          <div className="max-w-5xl mx-auto px-6 text-center">
              <div className="flex items-center justify-center gap-2 mb-4 opacity-50">
                  <ShieldCheck className="w-5 h-5 text-gray-600" />
                  <span className="font-bold text-gray-600">FactCheck AI</span>
              </div>
              <p className="text-gray-400 text-sm mb-6">
                  © 2025 FactCheck AI. Built as a college hackathon project.
              </p>
              <p className="text-gray-300 text-xs max-w-md mx-auto">
                  Disclaimer: The system provides AI-assisted predictions based on available data patterns and does not guarantee absolute accuracy. Always verify important information through multiple sources.
              </p>
          </div>
      </footer>

    </div>
  );
}