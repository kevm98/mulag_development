%% Version 2.8
%%
%% LaTeX2e Formatklasse für Diplom- oder Studienarbeit    -*-Mode: LaTeX;-*-
%% Autor: Roland Bless 1996-2001 bless@telematik.informatik.uni-karlsruhe.de
%% angepasst für AG Robotersysteme, Uni Kaiserslautern von Tobias Luksch,
%%                            luksch@informatik.uni-kl.de
%% --
%% (falls die Umlaute nicht richtig dargestellt werden: 
%%  der benutzte Zeichensatz ist utf8!)
%% Hinweis für Literatur zu LaTeX 2e:
%% 1. Leslie Lamport: "Das LaTeX-Handbuch", Addison-Wesley 1995
%% 2. Goosens, Mittelbach, Samarin: "Der LaTeX-Begleiter"
%% Referenz und Grundbefehle sind in 1. zu finden, 2. erklärt
%% Zusatz-Pakete, die einem Dokument ein individuelleres Aussehen verleihen
%% können.
%%
%% Zusatzpakete und LaTeX können von ftp://ftp.dante.de bezogen werden
%% Fragen zu LaTeX: siehe FAQ auf http://www.dante.de
%%
%% $Id: wissdoc.cls,v 1.1 2001/01/24 19:58:37 bless Exp bless $
%%
\NeedsTeXFormat{LaTeX2e}[1995/12/01] % Sorry, wir brauchen LaTeX2e!
\ProvidesClass{template/rrlab}[2012/01/10 Jochen Hirth, Daniel Schmidt und Tobias Luksch - Wissenschaftliches Dokument]

\newcommand{\langpath}{template/en}

% Options:
% boldauthor - hebt den in der Datei 'BoldAuthor.bib' definierten Autor im Literaturverzeichnis hervor.
% colorlinks - empfohlen für Bildschirmpräsentation, kann verwendet werden um beim Bauen mit 'pdflatex' die farbigen Links einzuschalten
% draft - Läßt Titelseite, Index, etc. weg., erzeugt Fußzeile mit Zusatzinformationen
%         Dafür muß allerdings die Unterstützung in der LaTeX Hauptdatei
%         vorhanden sein (\ifnotdraft{\include{bla...}}).
% showlabels - Zeigt alle LaTeX-Labels an der verwendeten Stelle an [EXPERIMENTAL]
% de,en - setzt die Vorlage auf englisch bzw. deutsch. default ist englisch
% diss - zum Erstellen einer Diss zur Abgabe im Fachbereich
% dissfinal - zum Erstellen einer Diss zur Abgabe beim Verlag
% latin1 - setzt Input Encoding auf latin1, standardmäßig utf8
% relaxed - erlaubt etwas unschönere Abstände für automatische Zeilenumbrüche
% rgbcolor - empfohlen für Bildschirmpräsentation, setzt die Farbkodierung auf rgb,standardmäßig cmyk
% icsecondpage - 'image correction for 2nd page' mach für die 2. Seite eine Bindekorrektur die der, der roten Deckblätter für Diplom-,
%                 Bachelor- und Masterarbeiten entspricht. Dies ist hilfreich, wenn man den Titel der Arbeit nicht auf das Deckblatt
%                drucken will (dafür gibt es ebenfalls eine Vorlage), sondern das Titelfeld ausschneiden möchte. Dann passt der Titel
%                auf der 2. Seite genau in dieses Fenster.
% thesis (previously dipl) - Da bei Diplomarbeiten keine zwei Gutachter benötigt werden, kann man mit dieser Option das Erscheinen des 2. Gutachters ausschalten.
% external Bietet die Möglichkeit mit \RRLABinstitution{x} die Institution - Name der Uni, Arbeitsgruppe etc - manuell zu setzten.

\let\@draft\@undefined
\DeclareOption{draft}{
  \PassOptionsToClass{draft}{book}
  \let\@draft\@empty
}
\let\@showlabels\@undefined
\DeclareOption{showlabels}{
  \let\@showlabels\@empty
}
\let\@colorlinks\@undefined
\DeclareOption{colorlinks}{\let\@colorlinks\@empty}
\newif\if@usegerman \@usegermanfalse
\DeclareOption{en}{}
\DeclareOption{de}{\renewcommand{\langpath}{template/de} \@usegermantrue}
\DeclareOption{relaxed}{\tolerance=1500 \emergencystretch=12pt}
\newif\if@useutf\@useutftrue
\DeclareOption{latin1}{\@useutffalse}
\newif\if@useboldauthor \@useboldauthorfalse
\DeclareOption{boldauthor}{\@useboldauthortrue}
\newif\if@usecmyk\@usecmyktrue
\DeclareOption{rgbcolor}{\@usecmykfalse}
\DeclareOption{icsecondpage}{\let\@icsecondpage\@empty}
\DeclareOption{dissfinal}{\let\@dissfinal\@empty}
\DeclareOption{diss}{\let\@diss\@empty}
\DeclareOption{dipl}{\let\@thesis\@empty}
\DeclareOption{thesis}{\let\@thesis\@empty}
\DeclareOption{summary}{\let\@summary\@empty}
\newif\if@report\@reportfalse
\DeclareOption{report}{\@reporttrue}
\let\@external\@undefined
\DeclareOption{external}{
  \let\@external\@empty
}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessOptions


%% Load latex class
\LoadClass[a4paper,12pt,titlepage]{book} % diese Klasse basiert auf ``book''

%% Include packages
\usepackage[T1]{fontenc}        % T1-encoded fonts: auch Wörter mit Umlauten trennen
\usepackage{lmodern}            % Neuerer Ersatz für Schriftfamilie 'ae' (zusammen mit fontenc)
\if@useutf
  \usepackage[utf8x]{inputenc}  % Eingabe nach utf8
\else
  \usepackage[latin1]{inputenc} % Eingabe nach ISO 8859-1 (Latin1)
\fi
\usepackage[final]{graphicx}    % um Graphiken einzubinden
\usepackage{makeidx}            % wir wollen auch einen Index
\usepackage{geometry}           % Seitenränder einstellen leichtgemacht
\usepackage{fancyhdr}           % definiere einfache Headings (mindestens V 1.99c
                                % notwendig)
\usepackage{longtable}          % seitenübergreifende Tabellen
\usepackage{booktabs}           % fuer spezielle unterteilungslinien
\usepackage[T1]{url}            % zum Darstellen von URLs, zusätzlich werden Zeilenumbrüche in URLs ermöglicht
\usepackage[labelfont={small,bf},textfont={small}]{caption}[2008/08/24] % die Schriftgröße für Captions ist small,
                                % die Labels sind zusätzlich fett
\usepackage{subcaption}         % für subfigure environment (Ersatz für subfig)
\usepackage[tight,nice]{units}  % Typografisch korrekte Darstellung von Einheiten
\usepackage{listings}           % Einbindung und Darstellung von Quellcode
\usepackage{microtype}          % Typografische Erweiterungen
\if@usecmyk
  \usepackage[cmyk]{xcolor}     % Farbkodierung für Druck (CMYK)
\else
  \usepackage[rgb]{xcolor}           % Farbkodierung für Bildschirm (RGB)
\fi
\usepackage{amssymb,amsmath,dsfont} % Mathematikumgebung und -symbole
\usepackage{ifthen}             % ifthenelse Syntax
\usepackage{hyperref}             % reference links

%% ---------------- end of usepackages -------------

\setcounter{secnumdepth}{3} % Numerierung auch für \subsubsection
\setcounter{tocdepth}{3}    % nimm auch \subsubsections ins Inhaltsverz. auf

%% Schusterjungen und Hurenkinder verhindern.
%% Siehe deutsches TeX-FAQ (6.1.3)
\clubpenalty = 2000
\widowpenalty = 2000
\displaywidowpenalty = 2000

%% New Commands
\newcommand{\ifnotdraft}[1]{\ifx\@draft\@undefined #1 \fi}
\newcommand{\ifdraft}[1]{\ifx\@draft\@empty #1 \fi}
\newcommand{\ifnotdraftelse}[2]{\ifx\@draft\@undefined #1 \else #2 \fi}
\newcommand{\ifshowlabels}[1]{\ifx\@showlabels\@empty #1 \fi}
\newcommand{\ifcolorlinks}[1]{\ifx\@colorlinks\@empty #1 \fi}
\newcommand{\ifdissfinal}[1]{\ifx\@dissfinal\@empty #1 \fi}
\newcommand{\ifnotdissfinal}[1]{\ifx\@dissfinal\@undefined #1 \fi}
\newcommand{\ifnotdissfinalelse}[2]{\ifx\@dissfinal\@undefined #1 \else #2 \fi}
\newcommand{\ifnotdiss}[1]{\ifx\@diss\@undefined #1 \fi}
\newcommand{\ifnotdisselse}[2]{\ifx\@diss\@undefined #1 \else #2 \fi}
\newcommand{\ifnotsummaryelse}[2]{\ifx\@summary\@undefined #1 \else #2 \fi}
\newcommand{\ifnotsummary}[1]{\ifx\@summary\@undefined #1 \fi}
\newcommand{\ifsummary}[1]{\ifx\@summary\@empty #1 \fi}
\newcommand{\ificsecondpage}[1]{\ifx\@icsecondpage\@empty #1 \fi}
\newcommand{\ifthesis}[2]{\ifx\@thesis\@empty #1 \else #2 \fi}
\newcommand{\ifdipl}[2]{\ifthesis{#1}{#2}}
\newcommand{\ifusegerman}[2]{\if@usegerman #1 \else #2 \fi}
\newcommand{\ifnotexternal}[1]{\ifx\@external\@undefined #1 \fi}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% check if we have to include ngerman package and also changes other language dependent stuff
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\if@usegerman
  \renewcommand{\bibname}{Literaturverzeichnis}
  \usepackage[algoruled,linesnumbered,german,algochapter]{algorithm2e}
  \RequirePackage{ngerman}[1998/07/08] % (V2.5e) Neue Rechtschreibung!
\else
 \renewcommand{\bibname}{Bibliography} 
 \usepackage[algoruled,linesnumbered,algochapter]{algorithm2e}
\fi


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setting algorithm and verbatim fonts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\SetAlFnt{\small} % Schriftgröße small für Algorithmen
\SetAlCapFnt{\small} % Schriftgröße small für Algorithmen-Captions
\def\verbatim@font{\small} % Schriftgröße small für Verbatim-Umgebung


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PDF dependent stuff (inspired by Frank Pählke)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newif\if@usepdf \@usepdffalse % not pdf ready
\RequirePackage{ifpdf}
\ifpdf
    \@usepdftrue % switch automatically to PDF mode
\else
    \ifx\pdftexversion\@undefined % kein PDFTeX aktiv
        \newcommand\pdfinfo[1]{}
    \fi
\fi


   \definecolor{rltred}{rgb}{0.6,0,0}
   \definecolor{rltgreen}{rgb}{0,0.5,0}
   \definecolor{rltblue}{rgb}{0,0,0.75}
   \definecolor{Gray}{gray}{0.5}
\if@usepdf
   \if@usegerman
     \PassOptionsToPackage{pdftex,
      \ifdissfinal{
       hidelinks,
      }
      \ifcolorlinks{
       colorlinks=true,%        % true for colored links
       }
       urlcolor=rltblue,%       % \href{...}{...} external (URL)
       filecolor=rltgreen,%     % \href{...} local file
       linkcolor=rltred,%       % \ref{...} and \pageref{...}
       citecolor=rltgreen,%
       bookmarks=true,%
       bookmarksopenlevel=1,%
       bookmarksopen=true,%
       bookmarksnumbered=true,%
       hyperindex=true,% 
       plainpages=false,% correct hyperlinks
       pdfpagelabels=true,% view TeX pagenumber in PDF reader
       ngerman
      }{hyperref} % erzeuge Hyperlinks z.B. für pdflatex
   \else
     \PassOptionsToPackage{pdftex,
       \ifdissfinal{
        hidelinks,
       }
       \ifcolorlinks{
       colorlinks=true,%        % true for colored links
       }
       urlcolor=rltblue,%       % \href{...}{...} external (URL)
       filecolor=rltgreen,%     % \href{...} local file
       linkcolor=rltred,%       % \ref{...} and \pageref{...}
       citecolor=rltgreen,%
       bookmarks=true,%
       bookmarksopenlevel=1,%
       bookmarksopen=true,%
       bookmarksnumbered=true,%
       hyperindex=true,% 
       plainpages=false,% correct hyperlinks
       pdfpagelabels=true% view TeX pagenumber in PDF reader
     }{hyperref} % erzeuge Hyperlinks z.B. für pdflatex
   \fi
%    \usepackage{thumbpdf}
 \pdfcompresslevel=9   % scheint zwar nichts zu bringen, aber was solls
\fi

\newif\if@usehyperref
\ifx\href\@undefined % Wenn kein hyperref Paket geladen
\@usehyperreffalse % hyperref used
\else
\@usehyperreftrue % hyperref used
\fi

\if@usehyperref
\newcommand{\urltext}[1]{\href{#1}{\url{#1}}} % URL die im Text ausgegeben wird.
\else
% Einige Dummy-Makros als Ersatz für das hyperref-Paket:
\newcommand{\href}[2]{#2}
\newcommand{\hyperbaseurl}[1]{}
\newcommand{\hyperimage}[1]{[IMAGE: #1]}
\newcommand{\hyperdef}[2]{}
\newcommand{\hyperref}[4]{#4}
\newcommand{\hyperlink}[2]{#2}
\newcommand{\hypertarget}[2]{#2}
\let\pdfbookmark\@gobbletwo
\newcommand{\urltext}[1]{\url{#1}}
\fi

% Das float-Paket muss NACH hyperref eingebunden werden,
% sonst klappt die PDF-Ausgabe nicht:
%\usepackage{float} %z.B. \floatstyle{ruled}

%% Seitengröße mit geometry-Paket einstellen
\geometry{includehead,includefoot,paper=a4paper,inner=2.5cm,outer=2.5cm,top=1.5cm,bottom=1.5cm,headheight=6mm,headsep=7mm,footskip=15mm}

%% Stil
\parindent 0cm                     % Absatzanfang wird nicht eingerückt
\parskip1.0ex plus0.5ex minus0.5ex % Abstand zwischen zwei Absätzen

\newcommand{\headfont}{\slshape}
\newcommand{\chapterheadfont}{}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Erweiterter Zugriff auf Datums- und Zeitwerte
%% (Achtung: werden bei Verwendung von 'babel' überschrieben)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifusegerman{%
  \usepackage[german]{datetime}
  \newdateformat{currentdate}{\twodigit{\THEDAY}. \monthnamegerman[\THEMONTH] \THEYEAR}      % Standardformat
  \newdateformat{versiondate}{\twodigit{\THEDAY}-\shortmonthnameenglish[\THEMONTH]-\THEYEAR} % Format für draft-Fußzeile
  }{%
  \usepackage{datetime}
  \newdateformat{currentdate}{\monthname[\THEMONTH] \twodigit{\THEDAY}, \THEYEAR}            % Standardformat
  \newdateformat{versiondate}{\twodigit{\THEDAY}-\shortmonthname[\THEMONTH]-\THEYEAR}        % Format für draft-Fußzeile
  }

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Headings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Geaenderte Headings:
%% Seitenzahl normal, Text in slanted und nicht in Grossbuchstaben
%%
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{\thechapter.\ #1}{}}
\fancyhf{} % clear all header and footer fields
\fancyhead[LE,RO]{{\headfont\thepage}} % left/right header for even/odd pages
\fancyhead[LO]{\headfont\nouppercase{\rightmark}} % header for left side (odd)
\fancyhead[RE]{\headfont\nouppercase{\leftmark}} % right header for even pages
\renewcommand{\headrulewidth}{0.5pt} % head rule

%% Versionsangabe bei gesetzter draft-Option
%%
\ifnotdraftelse{%
  \renewcommand{\footrulewidth}{0.0pt} % no rule
  }{%
  \fancyfoot[LE,LO]{\texttt{\scriptsize page:{\thepage} \hspace{10mm} revision:{\input{revision}} }}
  \fancyfoot[CE,CO]{\texttt{\scriptsize author:{\RRLABVARauthor} }}  %file:{\jobname}.tex
  \fancyfoot[RE,RO]{\texttt{\scriptsize date/time:{\versiondate\today}/{\currenttime}}}
  \renewcommand{\footrulewidth}{0.0pt} % no rule
} 

%% plainstyle
%%
\fancypagestyle{plain}{%
\fancyhf{} % clear all header and footer fields
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
%% Versionsangabe bei gesetzter draft-Option auch auf 'plain' Seiten
\ifnotdraftelse{}{%
  \fancyfoot[LE,LO]{\texttt{\scriptsize page:{\thepage} \hspace{10mm} revision:{\input{revision}} }}
  \fancyfoot[CE,CO]{\texttt{\scriptsize author:{\RRLABVARauthor} }}
  \fancyfoot[RE,RO]{\texttt{\scriptsize date/time:{\versiondate\today}/{\currenttime}}}
  \renewcommand{\footrulewidth}{0.0pt} % no rule
} 
}

\newif\if@nochapters\@nochaptersfalse
\if@report
\@nochapterstrue
\fi
\ifsummary{\@nochapterstrue}
\if@nochapters
\pagestyle{empty}
\renewcommand\thesection{\arabic{section}}
\renewcommand\thesubsection{\thesection.\arabic{subsection}}
\renewcommand{\chapter}[2]{
\ifthenelse{\equal{#1}{*}}{\section*{#2}}{\section{#1}}
}
\fi

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Stil der Ueberschriften
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewcommand{\chaptername}{}

\renewcommand{\section}{%
\@startsection{section}%
{1}% Gliederungsebene
{0mm}% Einzug
{2ex plus 1ex minus 0.5ex}% Vorabstand
{1.0ex plus 0.5ex minus 0.3ex}% Nachabstand
{\chapterheadfont\Large\bfseries}% Stil
}
\renewcommand{\subsection}{%
\@startsection{subsection}%
{2}%
{0mm}%
{2ex plus 1ex minus 0.5ex}%
{0.7ex plus 0.5ex minus 0.0ex}%
{\chapterheadfont\large\bfseries}%
}
\renewcommand{\subsubsection}{%
\@startsection{subsubsection}%
{3}%
{0mm}%
{2ex plus 1ex minus 1ex}%
{0.7ex plus 0.3ex minus 0.0ex}%
{\chapterheadfont\normalsize\bfseries}%
}
\renewcommand{\paragraph}{%
\@startsection{paragraph}%
{4}%
{0mm}%
{2ex plus 1ex minus 1ex}%
{0.5ex plus 0.3ex minus 0.0ex}%
{\chapterheadfont\normalsize\bfseries}%
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Stil der Kapitelueberschrift 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\newlength{\chapnolen}
\newlength{\chapparlen}
\newsavebox{\chapno}
%% Kein ``Kapitel'' bei Kapitelüberschrift
% \renewcommand{\@chapapp}{} allein reicht nicht!
%% \chapter{...}
\renewcommand{\@makechapterhead}[1]{%
  \vspace*{0.2\textheight}%
  %\hrule
  \vskip 15\p@
  {\parindent \z@ \raggedright \normalfont
    \ifnum \c@secnumdepth >\m@ne
      \if@mainmatter
        \savebox{\chapno}{\chapterheadfont\huge\bfseries \thechapter.}
        \settowidth{\chapnolen}{\usebox{\chapno}}
        \parbox[t]{\chapnolen}{\usebox{\chapno}}\nobreak\leavevmode
      \fi
    \fi
    \interlinepenalty\@MM
    \setlength{\chapparlen}{\textwidth}
    \addtolength{\chapparlen}{-1.0\chapnolen}
    \addtolength{\chapparlen}{-2ex}
    \leavevmode\nobreak
    \parbox[t]{\chapparlen}{\raggedright\chapterheadfont\huge \bfseries #1\par\nobreak}
    \vskip 40\p@
  }}

%% \chapter*{...}
\renewcommand{\@makeschapterhead}[1]{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \normalfont
    \interlinepenalty\@M
    \chapterheadfont \huge \bfseries  #1\par\nobreak
    \vskip 40\p@
  }%
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Inhaltsverzeichnis (auch andere Fontfamilie)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\let\oldtableofcontents\tableofcontents
\if@usepdf
 \renewcommand{\tableofcontents}{{\pdfbookmark{\contentsname}{\contentsname}\chapterheadfont\addtolength{\parskip}{-4pt}\oldtableofcontents}}
\fi
\let\@olddottedtocline\@dottedtocline
\renewcommand{\@dottedtocline}[5]{\@olddottedtocline{#1}{#2}{#3}{#4}{\chapterheadfont #5}}

%% pagestyle fuer Titelseite:
\def\titlehead#1{\def\@titlehead{#1}}
\def\titlefoot#1{\def\@titlefoot{#1}}
\def\@titlehead{}
\def\@titlefoot{}

\def\ps@title{%
\headheight 15mm
\footskip   0cm
\addtolength{\textheight}{-15mm}
\let\@mkboth\@gobbletwo%
  \def\@oddhead{\vbox{\hbox to\textwidth{\@titlehead}
                \vskip 1.5mm
                \hbox to\textwidth{\hrulefill}}}
  \def\@oddfoot{\vbox{\hbox to\textwidth{\hrulefill}
                \hbox to\textwidth{\@titlefoot}}}
  \let\@evenhead\@oddhead
  \let\@evenfoot\@oddfoot
}

\renewenvironment{titlepage}
{%
  \let\oldheadheight\headheight
  \let\oldfootskip\footskip
  \let\oldtextheight\textheight
  
   \cleardoublepage
   \if@twocolumn
      \@restonecoltrue\onecolumn
   \else
      \@restonecolfalse\newpage
    \fi
    \thispagestyle{title}%
    \setcounter{page}\@ne
}%
{\if@restonecol\twocolumn \else \newpage \fi
 \if@twoside\else
    \setcounter{page}\@ne
 \fi
 \let\headheight\oldheadheight
 \let\textheight\oldtextheight
 \let\footskip\oldfootskip
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Stil Index-Seite
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\renewenvironment{theindex}
{\if@twocolumn
   \@restonecolfalse
 \else
   \@restonecoltrue
 \fi
 \columnseprule \z@
 \columnsep 35\p@
 \twocolumn[\@makeschapterhead{\indexname}]%
 \@mkboth{\indexname}%
         {\indexname}%
 \thispagestyle{fancy}\parindent\z@
 \parskip\z@ \@plus .3\p@\relax
 \let\item\@idxitem
 \raggedright}

\newtheorem{definition}{Definition}[chapter]
\newtheorem{satz}{Satz}[chapter]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Makros fuer Dokumentaufbau (T.L.)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Möglichkeit die Institution manuell zu setzen (fuer arbeiten von externen Studenten)
\newcommand{\RRLABinstitution}[1]{
  \newcommand{\RRLABVARinstitution}{#1}
}
\ifnotexternal{
 \if@usegerman
  \RRLABinstitution{Arbeitsgruppe Robotersysteme\\ Fachbereich Informatik\\ Rheinland-Pf\"alzische Technische Universit\"at Kaiserslautern-Landau}
 \else
  \RRLABinstitution{Robotics Research Lab\\ Department of Computer Science\\ University of Kaiserslautern-Landau}
 \fi
}

% Titel der Arbeit
\newcommand{\RRLABtitle}[1]{
  \ifpdf
    \if@usehyperref
      \hypersetup{pdftitle={#1}} 
    \fi
  \fi
  \newcommand{\RRLABVARtitle}{#1}}
% Autor der Arbeit
\newcommand{\RRLABauthor}[1]{
  \ifpdf
    \if@usehyperref
      \hypersetup{pdfauthor={#1}} 
    \fi
  \fi
  \newcommand{\RRLABVARauthor}{#1}}
% Typ der Arbeit, Master Thesis, Dissertation, ...
\newcommand{\RRLABtype}[1]{\newcommand{\RRLABVARtype}{#1}}
% Datum der Ausgabe der Arbeit
\newcommand{\RRLABinception}[1]{\newcommand{\RRLABVARinception}{#1}}
% Datum der Abgabe der Arbeit
\newcommand{\RRLABsubmission}[1]{\newcommand{\RRLABVARsubmission}{#1}}
% Erster Gutachter
\newcommand{\RRLABfirstreviewer}[1]{\newcommand{\RRLABVARfirstreviewer}{#1}}
% Zweiter Gutachter
\newcommand{\RRLABsecondreviewer}[1]{\newcommand{\RRLABVARsecondreviewer}{#1}}

%% Nur für Projekt-, Diplomarbeit, Bachelor-, Masterthesis
% Betreuer
\newcommand{\RRLABsupervisor}[1]{\newcommand{\RRLABVARsupervisor}{#1}}

%% Nur für Dissertationen
% Angestrebter Titel
\newcommand{\RRLABdegree}[1]{\newcommand{\RRLABVARdegree}{#1}}
% Datum der Abgabe der Aussprache
\newcommand{\RRLABdefense}[1]{\newcommand{\RRLABVARdefense}{#1}}
% Vorsitzender
\newcommand{\RRLABchair}[1]{\newcommand{\RRLABVARchair}{#1}}
% Dekan
\newcommand{\RRLABdean}[1]{\newcommand{\RRLABVARdean}{#1}}


\newcommand{\RRLABtitlepage}[2]{
  \ifnotdraft{
    \pagenumbering{alph}
    \ifnotdissfinalelse{
      \newcommand{\RRLABVARtitledate}{#2}
      \ifnotdisselse{
        \ifnotsummaryelse{
            \newcommand{\RRLABVARtitleimage}{#1}
            \include{\langpath/rrlab_titlepage}
        }
        {
            \include{template/diss/rrlab_summary_titlepage}
        }
      }
      {
        \include{template/diss/rrlab_diss_titlepage}
      }
      \ifnotsummary{
       \clearpage{\pagestyle{empty}\cleardoublepage}
      }
      \ifnotsummaryelse{
       \pagenumbering{roman}
      }
      {
       \pagenumbering{arabic}
      }
      \setcounter{page}{1}
    }
    {
      \include{template/diss/rrlab_diss_final_titlepage}
    }
  }
}


\newcommand{\RRLABsecondpage}{
  \ifnotdraft{
    \ifnotdiss{
      \ifnotdissfinalelse{
        \include{\langpath/rrlab_secondpage}
        \clearpage{\pagestyle{empty}\cleardoublepage}
      }
      {
        \include{template/diss/rrlab_diss_final_imprint}
        \pagenumbering{roman}
        \setcounter{page}{1}
      }
    }
  }
}

%renew commands to adjust style of table of content for the summary (remove page numbers, dots and hyperrefs)
\ifsummary{
    \renewcommand*{\contentsline}[4]{
    \csname l@#1\endcsname{#2}{#3}
    }

    \renewcommand*\l@part[2]{%
    \ifnum \c@tocdepth >\m@ne
        \addpenalty{-\@highpenalty}%
        \vskip 1.0em \@plus\p@
        \setlength\@tempdima{1em}%
        \begingroup
        \parindent \z@ \rightskip \@pnumwidth
        \parfillskip -\@pnumwidth
        \leavevmode \bfseries
        \advance\leftskip\@tempdima
        \hskip -\leftskip
        #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss }\par
        \penalty\@highpenalty
        \endgroup
    \fi}

    \renewcommand*\l@chapter[2]{%
    \ifnum \c@tocdepth >\m@ne
        \addpenalty{-\@highpenalty}%
        \vskip 1.0em \@plus\p@
        \setlength\@tempdima{1.5em}%
        \begingroup
        \parindent \z@ \rightskip \@pnumwidth
        \parfillskip -\@pnumwidth
        \leavevmode \bfseries
        \advance\leftskip\@tempdima
        \hskip -\leftskip
        #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss }\par
        \penalty\@highpenalty
        \endgroup
    \fi}
    
    \newcommand*\summarytocline[5]{%
    \ifnum #1>\c@tocdepth \else
        \vskip \z@ \@plus.2\p@
        {\leftskip #2\relax \rightskip \@tocrmarg \parfillskip -\rightskip
        \parindent #2\relax\@afterindenttrue
        \interlinepenalty\@M
        \leavevmode
        \@tempdima #3\relax
        \advance\leftskip \@tempdima \null\nobreak\hskip -\leftskip
        {#4}\nobreak
        \leaders\hbox{$\m@th
            \mkern \@dotsep mu\hbox{}\mkern \@dotsep
            mu$}\hfill
        \nobreak
        \hb@xt@\@pnumwidth{\hfil\normalfont \normalcolor }%
        \par}%
    \fi}
    
    \renewcommand*\l@section{\summarytocline{1}{1.5em}{2.3em}}
    \renewcommand*\l@subsection{\summarytocline{2}{3.8em}{3.2em}}
    \renewcommand*\l@subsubsection{\summarytocline{3}{7.0em}{4.1em}}
    \renewcommand*\l@paragraph{\summarytocline{4}{10em}{5em}}
    \renewcommand*\l@subparagraph{\summarytocline{5}{12em}{6em}}
}

\newcommand{\RRLABthesistableofcontent}[1]{
  \ifsummary{
    \newpage
    \section*{Table of Content of the Thesis}
    \makeatletter
    \input{#1}
    \makeatother
  }
}

\newcommand{\RRLABdeclaration}[2]{
  \ifnotdraft{
    \ifnotdiss{
      \ifnotdissfinal{
        \newcommand{\RRLABVARdeclarationdate}{#1}
        \newcommand{\RRLABVARadmissiondate}{#2}
        \ifnotsummaryelse{
            \include{\langpath/rrlab_declaration}
            \clearpage{\pagestyle{empty}\cleardoublepage}
        }
        {
            \include{template/diss/rrlab_declaration_summary}
        }
      }
    }
  }
}


\newcommand{\RRLABpreface}[2]{
  \ifnotdraft{
    \chapter*{#1}
    #2
    \clearpage{\pagestyle{empty}\cleardoublepage}
  }
}

\newcommand{\RRLABabstract}[2]{
  \ifnotdraft{
    \chapter*{#1}
    #2
    \clearpage{\pagestyle{empty}\cleardoublepage}
  }
}


\newcommand{\RRLABcontents}{
  \ifdraft{
    \pagenumbering{roman}
   }
  \tableofcontents
  \cleardoublepage
  \setcounter{page}{1}
  \pagenumbering{arabic}
}


\newcommand{\RRLABindex}{
  \ifnotdraft{
    \cleardoublepage
    \if@usepdf
      \if@usehyperref
        \phantomsection %% nur falls pdf und hyperref benutzt werden
      \fi
    \fi
    \addcontentsline{toc}{chapter}{Index}
    \printindex
    \clearpage{\pagestyle{empty}\cleardoublepage}
  }
}


\newcommand{\RRLABmaketitle}{
 \vbox{\textbf{\Huge \RRLABVARtitle}}
 \vspace{1cm}
 {\Large \RRLABVARauthor}
 \vspace{2cm}
}

\newcommand{\RRLABpagenumbers}{
 \if@report
  \ClassWarning{rrlab}{setting pagestyle}
  \pagestyle{fancy}
  \fancyhf{}
  \cfoot{\thepage}
  \renewcommand{\headrulewidth}{0pt}
  \pagenumbering{arabic}
 \fi
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Definitionen fuer BibTex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\newcommand{\RRLABbibliography}[1]{
\if@report
\else
    \ifnotsummary{
        \cleardoublepage
    }
\fi
 \if@usepdf
    \if@usehyperref
     \phantomsection %% nur falls pdf und hyperref benutzt werden
    \fi
  \fi
 \ifnotsummary{
    \if@report
    \addcontentsline{toc}{section}{\bibname}
    \else
    \addcontentsline{toc}{chapter}{\bibname}
    \fi
 }
 \if@usegerman
  \bibliographystyle{template/bst/rrlab_german}
 \else
  \bibliographystyle{template/bst/rrlab_english}
 \fi
 \if@useboldauthor
  \nocite{BoldAuthor}
  \bibliography{#1,bold_author}
 \else
  \bibliography{#1}
 \fi
 \ifnotsummary{
    \clearpage{\cleardoublepage}
 }
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Anpassung der Labels wenn sie angezeigt werden
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\ifshowlabels{
  % safe original commands
  \let\labeloriginal\label
  % warning
  \newlinechar=`\^^J
  \typeout{LaTeX Warning: Option 'showlabels' is used which may cause errors with some packages (e.g. babel)!^^J}
  % create individual labels (one general and one for align environment)
  \renewcommand{\label}[1]{\labeloriginal{#1}\colorbox{yellow}{\scriptsize{\texttt{\url{#1}}}}\newline}
  \newcommand{\eqlabel}[1]{\labeloriginal{#1}\intertext{\colorbox{yellow}{\scriptsize{\texttt{\url{#1}}}}}\nonumber}
  % replace equation environment
  \renewenvironment{equation}{\align}{\endalign}
  % set label in align environment
  \def\label@in@display#1{%
    \ifx\df@label\@empty\else
      \@amsmath@err{Multiple \string\label's:
        label '\df@label' will be lost}\@eha
    \fi
      \eqlabel{#1}%\gdef\df@label{#1}%
  }
}


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Warnung bei Verwendung der veralteten Umgebung eqnarray
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\let\eqnarrayoriginal\eqnarray
\let\endeqnarrayoriginal\endeqnarray
\renewenvironment{eqnarray}{%
  \newlinechar=`\^^J
  \typeout{LaTeX Warning: Deprecated environment 'eqnarray' used on input line \the\inputlineno. (replace it by 'align' environment)^^J}
  \begin{eqnarrayoriginal}}{\end{eqnarrayoriginal}}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Noch einige Definitionen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Index-Datei öffnen
 \makeindex

\input{template/rrlab_macros}

%% **** END OF CLASS ****
