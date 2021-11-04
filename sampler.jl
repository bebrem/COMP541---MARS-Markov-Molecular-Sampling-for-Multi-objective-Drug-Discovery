using RDKitMinimalLib
using Logging
using TensorBoardLogger
using QuartzImageIO
using ImageMagick
include(rdkit.jl)
include(common/utils.jl)

module Sampler

    mutable struct sampler
        proposal::Function
        estimator::Function
        run_dir
        
        # for sampling
        step
        PATIENCE::Int
        patience
        best_eval_res::Float64 
        best_avg_score::Float64 
        last_avg_size
        train
        num_mols::Int
        num_step::Int
        log_every
        batch_size::Int
        score_wght
        score_succ
        score_clip
        fps_ref::String
        
        dataset 
        dataset_max_size::Int
        optimizer
        
        adv
        real_mols
        optimizer_adv   
    end

    function scores_from_dics(s::sampler,scores)
        scores = []
        score_norm = sum(s.score_wght)
        for score_dict in dicts
            score = 0.0
            v = values(score_dict)
            index = 1
            for k in keys(score_dict)
                if s.score_clip[k] > 0.0
                    v_current = min(v[index],s.score_clip[k])
                else
                    v_current = v[index]
                score += s.score_wght[k]*v_current
                end
            end
            score/=score_norm
            score = max(score,0.0)
            scores.append(score)
        end
    end


    function record(s::sampler,step, old_mols, old_dicts, acc_rates)
        old_scores = scores_from_dics(s,old_dicts)
        avg_scores = 1. * sum(old_scores)/len(old_scores)
        sizes = [GetNumAtoms(mol) for mol in old_mols]
        avg_size = sum(sizes) / len(old_mols)
        s.last_avg_size = avg_size
        
        ## succesful rate and uniqueness
        fps_mols = []
        unique = Set()
        success_dict = Dict(k => 0. for k in keys(old_dicts[1]))
        success, novelty, diversity = 0., 0., 0.
        
        for (i,score_dict) in enumerate(old_dicts)
            all_success = true
            for (j,(k,v)) in enumerate(score_dict)
                if (v >= s.score_succ[k])
                    success_dict[k] += 1
                else
                    all_success = false
                end
            end
            success += all_success
            if all_success
                fps_mols.append!(old_mols[i])
                unique = push!(get_smiles(old_mols[i]))
            end
        end
        success_dict = Dict(k => (v ./ length(old_mols)) for (i,(k,v)) in enumerate(success_dict))
        success = 1. * success / length(old_mols)
        unique = 1. * length(unique) / (length(fps_mols) + 1e-6)
        
        ### novelty and diversity
        fp_details = Dict{String, Any}("nBits" => 2048, "radius" => 3)
        fps_mols = [get_morgan_fp(x, fp_details) for x in fps_mols]
        if s.fps_ref
            n_sim = 0.0
            for i in range(1,length=length(fps_mols))
                sims = overlapMolecules(fps_mols[i], s.fps_ref)
                if maximum(sims) >= 0.4
                    n_sim += 1
                end
            end
            novelty = 1. - 1. * n_sim / (length(fps_mols) + 1e-6)
        else
            novelty = 1.0
        end
        
        similarity = 0.0
        for i in range(1,length=length(fps_mols))
            sims = [overlapMolecules(fps_mols[i], fps_mols[j]) for j in range(1,length=length(fps_mols))]
            similarity += sims
        end
        n = length(fps_mols)
        n_pairs = n * (n-1) / 2
        diversity = 1 - similarity / (n_pairs + 1e-6)
        
        diversity = minimum(diversity, 1.)
        novelty = minimum(novelty, 1.)
        evaluation = Dict("success"=>success,"unique"=> unique,"novelty"=> novelty,
            "diversity"=> diversity,"prod" => (success * novelty * diversity))
        
        
        ### logging and writing it to a file
        logger=TBLogger("Documents/Deep Learning/tensorboard_logs/run", min_level=Logging.Info)
        with_logger(logger) do
            @info scr = avg_score stp =  step 
            @info score_avg =  avg_score size_avg = avg_size stp = step
            @info succ_dic = success_dict stp = step
            @info eval = evaluation stp=step
            @info acc_rts = acc_rates  stp=step
            @info scores =  old_scores stp = step
            for k in keys(old_dicts[1])
                scores = [score_dict[k] for score_dict in old_dicts]
                @info kys =  scores stp = steps
            end
        end
        
        print_mols(s.run_dir, step, old_mols, old_scores, old_dicts)
        
        ### early stop
        if (evaluation["prod"] > 0.1 && evaluation["prod"] < s.best_eval_res  + 0.01 && avg_score > 0.1 && avg_score < s.best_avg_score + 0.01)
            s.patience -= 1
        else
            s.patience = s.PATIENCE
            s.best_eval_res  = max(s.best_eval_res, evaluation["prod"])
            s.best_avg_score = max(s.best_avg_score, avg_score)
        end        
    end

    function sample(s::Sampler,sun_dir,mols_init)
        #=
            sample molecules from initial ones
        @params:
            mols_init : initial molecules
        =#
        
        s.run_dir = run_dir
        
        ### sample
        old_mols = [mol for mol in mols_init]
        old_dicts = get_scores(old_mols) #estimator dan gelecek 
        old_scores = scores_from_dicts(old_dicts)
        acc_rates = [0.0 for i in old_mols]
        record(s,-1, old_mols, old_dicts, acc_rates)
        
        for step in range(1,length=s.num_step)
            if s.patience <= 0
                break
            end
            s.step = step
            new_mols, fixings = propose(old_mols) 
            new_dicts = get_scores(new_mols)
            new_scores = scores_from_dicts(new_dicts)
            
            indices = [i for i in range(1,length(old_mols)) if new_scores[i] > old_scores[i]]
            open(joinpath(run_dir,"edits.txt"),"a") do f 
                write(f,"edits at step $step")
                write(f,"improve act arm")
                for (i, item) in enumerate(proposaldataset):
                        _, edit = item
                        improve = new_scores[i] > old_scores[i]
                        write(f,"$improve $edit["act"] $edit["act"]")
                end
            end
            
            acc_rates = [min(1., max(0., A)) for A in acc_rates]
            for i in range(1,length=s.num_mols)
                A = acc_rates[i] # A = p(x') * g(x|x') / p(x) / g(x'|x)
                if rand() > A
                    continue
                end
                old_mols[i] = new_mols[i]
                old_scores[i] = new_scores[i]
                old_dicts[i] = new_dicts[i]
            end
            if step % s.log_every == 0
                record(step, old_mols, old_dicts, acc_rates)
            end
            
            ### train editor
            if s.train:
                dataset = proposaldataset
                dataset = data.Subset(dataset, indices)
                if self.dataset: 
                    self.dataset.merge_(dataset)
                else: self.dataset = ImitationDataset.reconstruct(dataset)
                n_sample = len(self.dataset)
                if n_sample > 2 * self.DATASET_MAX_SIZE:
                    indices = [i for i in range(n_sample)]
                    random.shuffle(indices)
                    indices = indices[:self.DATASET_MAX_SIZE]
                    self.dataset = data.Subset(self.dataset, indices)
                    self.dataset = ImitationDataset.reconstruct(self.dataset)
                batch_size = int(self.batch_size * 20 / self.last_avg_size)
                log.info('formed a imitation dataset of size %i' % len(self.dataset))
                loader = data.DataLoader(self.dataset,
                    batch_size=batch_size, shuffle=True,
                    collate_fn=ImitationDataset.collate_fn
                )
                            

        
        

        
        
    end 


end
