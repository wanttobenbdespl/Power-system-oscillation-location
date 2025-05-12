clear variables; clc;
global Source_V Amplitude_V Frequency_V vref0 ...
       Source_P1 Amplitude_P1 Frequency_P1 pm01 ...
       intstep VP_nos VQ_nos tcorr P0 Q0 PQ_std PQ Syn

%%% Careful! All Bus indices are shifted by 1. Therfore, "Bus 4" %%%
%%% (source Bus) is actually Bus #3 when calling its variables.  %%%

f = [0.24; 0.34; 0.41; 0.43; 0.60; 0.65; 0.69; 0.76; 0.77; 0.89; 0.93; 0.97; 0.99; 1.02; 1.12; 1.15; 1.23; 1.26; 1.29; 1.32; 1.36; 1.37; 1.45; 1.47; 1.50; 1.57; 1.72];
LBm = 0; RSLTm = 0.1; UBm = 0.45;
LBv = 0; RSLTv = 0.05; UBv = 0.225;

%% Initial Data
for xnode = 1:29
    for pqi = 0.95:0.05:1.05
        % 加载 WECC_179 数据
        try
            WECC_179;
        catch ME
            error('加载 WECC_179 文件失败: %s', ME.message);
        end


        npq = size(PQ.con, 1);
        P0 = PQ.con(:, 4);
        Q0 = PQ.con(:, 5);

        %% Solve Power Flow
        initpsat;
        datafile = 'WECC_179';
        runpsat(datafile, 'data');
        Settings.freq = 60;
        Settings.maxvar = 5000;
        runpsat('pf');
        
        %% Choose Excitation and Mechanical Oscillation Source Buses
        %  Sources are generator indices
        % Exciter Reference
        Source_V    = xnode;
        Amplitude_V = 0.015;
        Frequency_V = 0.86;
        vref0       = Exc.vref0;
        
        % Mechanical Power 2 (Generator Bus 65)
        Source_P1    = xnode;
        Amplitude_P1 = 0.5;
        Frequency_P1 = 0.7;
        pm01         = Syn.pm0(Source_P1);
        
        %% Pm Batch Simulation
        for Pf = 1:27
            Frequency_P1 = f(Pf);
            for Pmi = LBm:RSLTm:UBm
                Amplitude_P1 = 0.05;
                %% Initialize Time Domain Simulation
                intstep = 1/30;
                tcorr   = 1;
                PQ_std  = 0.01;
                tbegin  = 0;
                tfinal  = 30;
                Amplitude_P1 = Pmi + Amplitude_P1;
                runpsat('perturb_Pf','pert');
                Settings.freq   = 60;
                clpsat.readfile = 1;
                VP_nos  = zeros(npq,1);
                VQ_nos  = zeros(npq,1);
                
                % SETTINGS FOR TIME DOMAIN SIMULATION
                Settings.coi   = 0;
                Settings.t0    = tbegin;
                Settings.tf    = tfinal;
                Settings.pq2z  = 0;
                Settings.fixt  = 1;
                Settings.tstep = intstep;
                nL             = Line.n + Ltc.n + Phs.n + Hvdc.n + Lines.n;
                Varname.idx    = 1:DAE.n + DAE.m + 2*Bus.n + 6*nL + 2;
                
                % Run Time Domain Simulation
                runpsat('td');
                % Save The Result
                data.p=Varout.vars(:,[424;428;432;436;440;444;448;452;456;460;464;468;472;476;480;484;488;492;496;500;504;508;512;516;520;524;528;532;536]);
                data.p=data.p./data.p(1,:);
                data.q=Varout.vars(:,[425;429;433;437;441;445;449;453;457;461;465;469;473;477;481;485;489;493;497;501;505;509;513;517;521;525;529;533;537]);
                data.q=data.q./data.q(1,:);
                data.v=Varout.vars(:,[246;248;251;253;255;257;260;272;277;278;282;285;287;289;307;312;319;321;345;354;358;360;380;382;386;390;391;401;404]);
                [~,b]=find(Varout.vars(1,1:59)==1);
                dirName = strcat('sample_data/', num2str(xnode), '/');
                fileName = strcat(num2str(xnode), 'PQ_', num2str(pqi), 'Pf_', num2str(f(Pf)), '_', 'Pmi_', num2str(Pmi), '.mat');
                fullFilePath = [dirName, fileName];
 
                if ~exist(dirName, 'dir')
                    mkdir(dirName, 'recursive');
                end
 
                save(fullFilePath, 'data');
            end
        end
        %% Vref Batch Simulation
        for Vf = 1:27
            Frequency_V = f(Vf);
            for Vmi = LBv:RSLTv:UBv
                Amplitude_V = 0.025;
                %% Initialize Time Domain Simulation
                intstep = 1/30;
                tcorr   = 1;
                PQ_std  = 0.01;
                tbegin  = 0;
                tfinal  = 30;
                Amplitude_V = Vmi + Amplitude_V;
                runpsat('perturb_Verf','pert');
                Settings.freq   = 60;
                clpsat.readfile = 1;
                VP_nos  = zeros(npq,1);
                VQ_nos  = zeros(npq,1);
                
                % SETTINGS FOR TIME DOMAIN SIMULATION
                Settings.coi   = 0;
                Settings.t0    = tbegin;
                Settings.tf    = tfinal;
                Settings.pq2z  = 0;
                Settings.fixt  = 1;
                Settings.tstep = intstep;
                nL             = Line.n + Ltc.n + Phs.n + Hvdc.n + Lines.n;
                Varname.idx    = 1:DAE.n + DAE.m + 2*Bus.n + 6*nL + 2;
                
                % Run Time Domain Simulation
                runpsat('td');
                % Save The Result
                data.p=Varout.vars(:,[424;428;432;436;440;444;448;452;456;460;464;468;472;476;480;484;488;492;496;500;504;508;512;516;520;524;528;532;536]);
                data.p=data.p./data.p(1,:);
                data.q=Varout.vars(:,[425;429;433;437;441;445;449;453;457;461;465;469;473;477;481;485;489;493;497;501;505;509;513;517;521;525;529;533;537]);
                data.q=data.q./data.q(1,:);
                data.v=Varout.vars(:,[246;248;251;253;255;257;260;272;277;278;282;285;287;289;307;312;319;321;345;354;358;360;380;382;386;390;391;401;404]);
                [~,b]=find(Varout.vars(1,1:59)==1);
                data.w = Varout.vars(:,b);
                
                dirName = strcat('sample_data/', num2str(xnode), '/');
                fileName = strcat(num2str(xnode), 'PQ_', num2str(pqi), 'Vf_', num2str(f(Pf)), '_', 'Vmi_', num2str(Pmi), '.mat');
                fullFilePath = [dirName, fileName];
 
                if ~exist(dirName, 'dir')
                    mkdir(dirName, 'recursive');
                end
 
                save(fullFilePath, 'data');
            end
        end
        closepsat;
    end
end