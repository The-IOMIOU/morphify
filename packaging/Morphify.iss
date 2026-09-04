; Inno Setup script for Morphify.
;
; Build with:  tools\innosetup\ISCC.exe packaging\Morphify.iss
; or:          python packaging\build.py --installer
;
; Installs the one-folder PyInstaller output from dist\Morphify.
; Models are not shipped here — they are ~1.2 GB and the app downloads them
; on first launch into %LOCALAPPDATA%\Morphify\models.

#define AppName        "Morphify"
#define AppShortName   "Morphify"
#define AppVersion     "1.0.0"
#define AppPublisher   "Morphify"
#define AppURL         "https://github.com/hacksider/Deep-Live-Cam"
#define AppExe         "Morphify.exe"
#define SourceDir      "..\dist\Morphify"

[Setup]
AppId={{2F8B6D41-9C3A-4E15-B7D2-5A0E9C74F318}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
; Resolved in code: the bundle is ~3.5 GB, and the system drive is often the
; one without room for it. See PickInstallDrive().
DefaultDirName={code:DefaultInstallDir}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=..\dist
OutputBaseFilename={#AppShortName}-{#AppVersion}-Setup
SetupIconFile=Morphify.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; The bundle carries CUDA libraries and is 64-bit only.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; ~4 GB of payload needs plenty of headroom during extraction.
DiskSpanning=no
PrivilegesRequired=admin

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; \
    GroupDescription: "Additional shortcuts:"

[Files]
; The whole PyInstaller one-folder output, recursively.
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; PyInstaller writes __pycache__ next to the bundled modules at runtime.
Type: filesandordirs; Name: "{app}\_internal\__pycache__"

[Code]
const
  { DirectShow CLSID registered by the OBS Studio installer. Its presence is
    what lets the app publish a virtual camera, so warn early rather than
    letting the user discover a greyed-out button later. }
  OBSVirtualCamCLSID = '{{A3FCE0F5-3493-419F-958A-ABA1250EC20B}';
  { Bundle payload plus extraction headroom. }
  RequiredBytes = 4500000000;

var
  ErrorCode: Integer;

function DriveFreeBytes(const Drive: String): Int64;
var
  FreeAvailable, TotalSpace: Int64;
begin
  Result := -1;
  if GetSpaceOnDisk64(Drive, FreeAvailable, TotalSpace) then
    Result := FreeAvailable;
end;

function DefaultInstallDir(Param: String): String;
var
  SystemDir, Candidate, BestDrive: String;
  Index: Integer;
  Best, FreeSpace: Int64;
begin
  { Prefer the normal Program Files location when it actually fits. }
  SystemDir := ExpandConstant('{autopf}') + '\{#AppShortName}';
  if DriveFreeBytes(ExpandConstant('{autopf}')) >= RequiredBytes then
  begin
    Result := SystemDir;
    exit;
  end;

  { Otherwise offer the fixed drive with the most free space. Installing
    ~3.5 GB onto a full system drive fails late and confusingly; defaulting
    somewhere it fits is friendlier, and the user can still change it. }
  Best := -1;
  BestDrive := '';
  for Index := Ord('C') to Ord('Z') do
  begin
    Candidate := Chr(Index) + ':\';
    if DirExists(Candidate) then
    begin
      FreeSpace := DriveFreeBytes(Candidate);
      if FreeSpace > Best then
      begin
        Best := FreeSpace;
        BestDrive := Candidate;
      end;
    end;
  end;

  if (BestDrive <> '') and (Best >= RequiredBytes) then
    Result := BestDrive + '{#AppShortName}'
  else
    Result := SystemDir;
end;

function VirtualCameraInstalled(): Boolean;
begin
  Result := RegKeyExists(HKEY_CLASSES_ROOT,
    'CLSID\' + OBSVirtualCamCLSID + '\InprocServer32');
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    if not VirtualCameraInstalled() then
    begin
      if MsgBox(
        'Morphify publishes your swapped webcam feed through the OBS '
        + 'Studio virtual camera driver, which does not appear to be '
        + 'installed.' + #13#10 + #13#10
        + 'Everything else works without it, but other apps (Discord, Zoom, '
        + 'Teams, your browser) will not see the swapped feed until it is '
        + 'installed. OBS itself never needs to be running.' + #13#10 + #13#10
        + 'Open the OBS Studio download page now?',
        mbConfirmation, MB_YESNO) = IDYES then
      begin
        ShellExecAsOriginalUser('open', 'https://obsproject.com/download',
          '', '', SW_SHOWNORMAL, ewNoWait, ErrorCode);
      end;
    end;
  end;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\Morphify');
    if DirExists(DataDir) then
    begin
      if MsgBox(
        'Also delete downloaded models, your face library and settings?'
        + #13#10 + #13#10 + DataDir + #13#10 + #13#10
        + 'Choose No to keep them for a future reinstall.',
        mbConfirmation, MB_YESNO) = IDYES then
      begin
        DelTree(DataDir, True, True, True);
      end;
    end;
  end;
end;
