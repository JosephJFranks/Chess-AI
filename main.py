import torch
from model import ChessCNN
from agent import ChessAI
from torch.utils.tensorboard import SummaryWriter
import datetime
import os
from utils import *
from pathlib import Path
import re
import random

def intro_ui(directory):
    while True:
        choice = input("Choose to train a new model [\"NEW\"], load and train a current model [\"LOAD\"], get a game from a model [\"WATCH\"] or play against a model [\"PLAY\"]: ")
        if choice.lower() == "new":
            model_name = input("Please input a model name: ")
                    
            # Initialize the model and the AI agent
            chess_ai = ChessAI(model)

            return model_name, chess_ai, 'new'

        if choice.lower() == "load":
            
            while True:
                list_saved_models(directory)
                model_choice = input("Pick one of the models to load: ")

                try:
                    # Load the checkpoint
                    checkpoint = torch.load(f'{directory}/{model_choice}.pth', weights_only=False)
                    
                    # Load state_dict if it's a full checkpoint with model state
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # If it's only the state_dict (weights), load directly
                        model.load_state_dict(checkpoint)
                    
                    # Initialize the ChessAI with the model
                    chess_ai = ChessAI(model)

                    return model_choice, chess_ai, 'load'
                
                except Exception as e:
                    print(f'That seemed to not have worked: {str(e)}')
                    print('---------')

        if choice.lower() == "watch":
            
            while True:
                list_saved_models(directory)
                model_choice = input("Pick one of the models to load: ")

                try:
                    # Load the checkpoint
                    checkpoint = torch.load(f'{directory}/{model_choice}.pth', weights_only=False)
                    
                    # Load state_dict if it's a full checkpoint with model state
                    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        # If it's only the state_dict (weights), load directly
                        model.load_state_dict(checkpoint)
                    
                    # Initialize the ChessAI with the model
                    chess_ai = ChessAI(model)

                    return model_choice, chess_ai, 'watch'
                
                except Exception as e:
                    print(f'That seemed to not have worked: {str(e)}')
                    print('---------')

        
        if choice.lower() == 'play':
            print('Not currently running, sorry')


if __name__ == "__main__":
    directory = "./models"
    number_of_legal_moves = 4096  # 64**2 to represent the starting and ending states
    model = ChessCNN(number_of_legal_moves)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    project_name = "chess_ai"
    model_name, chess_ai, option = intro_ui(directory)
    log_dir = f'logs/{project_name}/{model_name}/runs/{timestamp}'

    writer = SummaryWriter(log_dir)

    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    if option == 'watch':
        game_data = chess_ai.self_play()
        total_game_moves = game_data[2]

        white_moves = game_data[0][1]  # White's moves
        black_moves = game_data[1][1]  # Black's moves

        # Determine who moved first and merge the moves accordingly
        for i in range(min(len(white_moves), len(black_moves))):
            total_game_moves.append(white_moves[i])
            total_game_moves.append(black_moves[i])

        # If White has an extra move (i.e., odd number of moves in total)
        if len(white_moves) > len(black_moves):
            total_game_moves.append(white_moves[-1])

        # If Black has an extra move (can happen in some cases)
        elif len(black_moves) > len(white_moves):
            total_game_moves.append(black_moves[-1])

        save_game_to_pgn(total_game_moves)

    # Train the model via self-play
    else:
        if option == 'new':
            os.makedirs(f"{directory}/{model_name}", exist_ok=True)
            num_episodes = 50
            latest_epoch = 0
        if option == 'load':
            while True:
                try:
                    additional_episodes = int(input("How many epochs more do you want to train for?" ))
                    break
                except Exception as e:
                    print(f'That seemed to not have worked: {str(e)}')
                    print('---------')
            latest_epoch = get_latest_epoch(model_name) + 11
            num_episodes = latest_epoch + additional_episodes

        for episode in range(latest_epoch, num_episodes):
            game_data = chess_ai.self_play()

            # Train both White and Black with their respective data
            white_loss = chess_ai.train_model(data=game_data[0], model=chess_ai.model, optimiser=chess_ai.optimiser, factor = 1)
            black_loss = chess_ai.train_model(data=game_data[1], model=chess_ai.model, optimiser=chess_ai.optimiser, factor = -1)

            total_loss = white_loss + black_loss

            print('Epoch:',  episode, '; Game Loss:', white_loss, ', ', black_loss)

            if episode % 10 == 0:
                torch.save({
                    'epoch': episode,
                    'model_state_dict': chess_ai.model.state_dict(),
                    'optimizer_state_dict': chess_ai.optimiser.state_dict(),
                }, f'models/{model_name}/{model_name}_epoch_{episode}.pth')

                print('Model saved')

            torch.save({
                'epoch': num_episodes,
                'model_state_dict': chess_ai.model.state_dict(),
                'optimizer_state_dict': chess_ai.optimiser.state_dict(),
            }, f'models/{model_name}/{model_name}_epoch_{episode}.pth')
    
        torch.save({
                    'model_state_dict': chess_ai.model.state_dict(),
                }, f'models/{model_name}.pth')
        print('Final Model Saved')

        writer.close()